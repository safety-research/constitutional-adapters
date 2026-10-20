"""scripts/sdfbase/slurm_fit_cell.py — the Slurm twin of the Modal app's
_fit_impl (2026-09-06). Pinned with a FAKE trainer (an executable that
receives exactly the trainer argv and writes what the real trainer would):

- the trainer invocation is the Modal wrapper's, flag for flag;
- preconditions refuse in milliseconds (rc 3) before any trainer run;
- the stale-COMPLETE supersede gate refuses (rc 4) without running;
- a publish under the expected fingerprint: rc 0, the reports/ tees, the
  transient checkpoint retained as final_ckpt/, the run record;
- a trainer that returns 0 but publishes another identity is rc 6; a
  failing trainer's rc propagates;
- the gates the app exposes (supersede_gate, retain_final_checkpoint,
  _marker_fingerprint) ARE the shared module's;
- the writable namespace is the CONFIG's (fit_cell_identity.out_key), never
  the pool's: two configs sharing a pool resolve to distinct roots, the
  wrapper refuses an --out-root that is another config's before a byte is
  written there, an in-flight frozen cell's checkpoint survives a pool-mate
  launch, and a fit already COMPLETE under another root of the bank is not
  refit from scratch (2026-09-10);
- the sample manifest is resolved and VALIDATED (cell, pool_sha256) through
  sisteer.data.sample_manifest — the trainer's resolver — so a stale or
  foreign manifest is refused before the trainer, and preflight and trainer
  name the same file or refuse for the same reason (2026-09-10).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _sub in ("scripts", "scripts/sdfbase"):
    _p = str(REPO / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fit_cell_identity as fci  # noqa: E402
import slurm_fit_cell as sfc  # noqa: E402
from sisteer.data.pairs import Pair, save_pairs  # noqa: E402

SI = "constv2-mix-1k"
ARM = "lora_r4_ce"
CELL = f"{SI}__{ARM}"
DRIVER = REPO / "scripts" / "sdfbase" / "slurm_fit_cell.py"

FAKE_TRAINER = r'''#!/usr/bin/env python
"""A stand-in for train_two_si_objects.py: argv = [trainer_path, "train",
flags...]. Writes the publication the real trainer would, under the
fingerprint FAKE_FP names; FAKE_MODE selects the behaviour."""
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[2:]
def flag(name, default=None):
    return args[args.index(name) + 1] if name in args else default
Path(os.environ["FAKE_CALLED"]).write_text(json.dumps(sys.argv))
mode = os.environ.get("FAKE_MODE", "publish")
if mode == "fail":
    print("[train-two-si] boom", flush=True); sys.exit(7)
out = Path(flag("--out")); si = flag("--si"); arm = flag("--arms")
cell = out / f"{si}__{arm}"
# the trainer refuses a config whose bytes differ from --expect-config-sha
cfg_sha = hashlib.sha256(Path(flag("--config")).read_bytes()).hexdigest()[:16]
assert flag("--expect-config-sha") == cfg_sha, (flag("--expect-config-sha"), cfg_sha)
if mode == "skip":
    print(f"[train-two-si] {si}__{arm}: validated COMPLETE — skipping (resume)", flush=True)
    sys.exit(0)
out.mkdir(parents=True, exist_ok=True)
(out / "_store_identity.json").write_text(json.dumps(
    {"model": "m", "model_revision": "r", "positions": flag("--positions")}))
cell.mkdir(parents=True, exist_ok=True)
(cell / "object.pt").write_bytes(b"OBJ")
(cell / "train_budget_meta.json").write_text("{}")
(cell / "metrics.jsonl").write_text("{}\n")
fp = os.environ["FAKE_FP"] if mode != "wrongfp" else "f" * 64
files = {n: hashlib.sha256((cell / n).read_bytes()).hexdigest()
         for n in ("object.pt", "train_budget_meta.json", "metrics.jsonl")}
(cell / "COMPLETE").write_text(json.dumps({
    "cell_id": cell.name, "si_id": si, "arm": arm, "positions": "all",
    "object_file": "object.pt", "object_sha256": files["object.pt"],
    "files": files, "fit_fingerprint": fp, "published_at": "now"}))
(out / f"{si}__fit_report.json").write_text(json.dumps(
    {"arms": {arm: {"status": "trained"}}}))
(out / f"{si}__metrics.jsonl").write_text("{}\n")
ck = Path(flag("--checkpoint-dir")) / cell.name
ck.mkdir(parents=True, exist_ok=True)
(ck / "ckpt.pt").write_bytes(b"CKPT")
print("[fitckpt] wrote checkpoint", flush=True)
print("done", flush=True)
'''


def _write_manifest(pairs: Path, *, cell: str = SI, spelling: str = "stem",
                    pool_sha: str | None = None, text: str | None = None) -> Path:
    """The manifest the pool builder writes beside ``pairs`` — its ``cell``
    and the pool FILE's sha256 (``pool_sha`` overrides, for a stale one);
    ``text`` writes raw bytes instead. ``spelling``: "stem" =
    pairs_<cell>.manifest.json (the builder's), "appended" =
    <pool>.jsonl.manifest.json."""
    path = (pairs.with_suffix(".manifest.json") if spelling == "stem"
            else Path(str(pairs) + ".manifest.json"))
    if text is not None:
        path.write_text(text)
        return path
    sha = pool_sha or hashlib.sha256(pairs.read_bytes()).hexdigest()
    path.write_text(json.dumps({"cell": cell, "budget": 2, "n_docs": 2,
                                "pool_sha256": sha}))
    return path


def _cfg(heldout: str) -> dict:
    return {
        "layer": 1, "max_len": 256, "data_mode": "documents",
        "training_budget_kind": "document_tokens", "completion_tokens": None,
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "model_revision": "0" * 40,
        "batch_size_by_arm": {ARM: 1},
        "train_kwargs": {
            "default": {"seed": 0, "holdout_frac": 0.0,
                        "heldout_pool": heldout, "probe_n": 2,
                        "early_stop_min_epochs": 1, "early_stop_patience": 2},
            ARM: {"lr": 1e-3, "lora_rank": 4, "lr_schedule": "constant",
                  "floor_ratio_max": None, "ce_floor_ratio_max": 1.1},
        },
        "train_kwargs_by_si": {SI: {"epochs": 2}},
        "si_ids": [SI],
    }


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A staged bank: config (declaring the /vol held-out path), the pool +
    manifest under $SISTEER_VOL_ROOT/pool/k/, the fake trainer, and the
    expected fingerprint through the shared derivation."""
    root = tmp_path / "vol"
    pool_dir = root / "pool" / "k"
    pool_dir.mkdir(parents=True)
    save_pairs([Pair(prompt="", pos="Held doc.", neg="", instruction_id=SI)],
               pool_dir / "pairs_constv2-heldout.jsonl")
    pairs = pool_dir / f"pairs_{SI}.jsonl"
    save_pairs([Pair(prompt="", pos="Train doc one.", neg="",
                     instruction_id=SI),
                Pair(prompt="S\n\n", pos="Train doc two.", neg="",
                     instruction_id=SI)], pairs)
    _write_manifest(pairs)          # the builder's spelling, cell + pool sha
    # named as a fits config is (configs/sdfbase_<key>_fits.yaml): the key
    # `k` is the config's identity and its namespace under the bank
    cfg = tmp_path / "sdfbase_k_fits.yaml"
    cfg.write_text(yaml.safe_dump(
        _cfg("/vol/pool/k/pairs_constv2-heldout.jsonl"), sort_keys=False))
    monkeypatch.setenv("SISTEER_VOL_ROOT", str(root))
    fake = tmp_path / "fake_trainer.py"
    fake.write_text(FAKE_TRAINER)
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    fp = fci.expected_fit_fingerprint(cfg, pairs, SI, ARM)
    called = tmp_path / "called.json"
    monkeypatch.setenv("FAKE_FP", fp)
    monkeypatch.setenv("FAKE_CALLED", str(called))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    out_root = tmp_path / "bank" / "k"
    return dict(cfg=cfg, pairs=pairs, fp=fp, fake=fake, called=called,
                out_root=out_root, root=root)


def _run(w, **over):
    argv = ["--config", str(over.pop("config", w["cfg"])), "--si", SI,
            "--arm", ARM, "--pairs", str(over.pop("pairs", w["pairs"])),
            "--out-root", str(over.pop("out_root", w["out_root"])),
            "--checkpoint-every", "5", "--python", str(w["fake"])]
    for k, v in over.items():
        argv += [k, v]
    return sfc.main(argv)


def _sibling_cfg(w, key: str = "k_r2") -> Path:
    """A SECOND config over the SAME pool — the `_full_r2t12` shape: the
    parent's bytes with one optimizer value changed — named as a fits config
    is, so its key is its identity."""
    d = yaml.safe_load(w["cfg"].read_text())
    d["train_kwargs"][ARM]["lr"] = 2e-3
    sib = w["cfg"].parent / f"sdfbase_{key}_fits.yaml"
    sib.write_text(yaml.safe_dump(d, sort_keys=False))
    return sib


def test_trainer_command_is_the_modal_wrappers(tmp_path):
    cmd = sfc.trainer_command("py", REPO, config="c.yaml", si=SI, arm=ARM,
                              pairs="p.jsonl", cells_dir=Path("/o/cells_all"),
                              expect_sha="ab" * 8, checkpoint_dir=Path("/o/ckpt"),
                              checkpoint_every=100, overwrite=False)
    assert cmd == ["py", str(REPO / "scripts" / "train_two_si_objects.py"),
                   "train", "--config", "c.yaml", "--si", SI, "--arms", ARM,
                   "--pairs", "p.jsonl", "--token-cache", "auto",
                   "--out", "/o/cells_all", "--expect-config-sha", "ab" * 8,
                   "--positions", "all", "--checkpoint-dir", "/o/ckpt",
                   "--checkpoint-every", "100"]
    # the same tokens, in the same order, as the Modal _fit_impl issues
    src = (REPO / "scripts" / "modal" / "sdfbase_fits.py").read_text()
    impl = src[src.index("def _fit_impl("):src.index("def _config_sha16(")]
    for tok in ('"train"', '"--config"', '"--si"', '"--arms"', '"--pairs"',
                '"--token-cache", "auto"', '"--out"', '"--expect-config-sha"',
                '"--positions", "all"', '"--checkpoint-dir"',
                '"--checkpoint-every"'):
        assert tok in impl, tok
    assert sfc.trainer_command("py", REPO, config="c", si=SI, arm=ARM,
                               pairs="p", cells_dir=Path("/o"),
                               expect_sha="x", checkpoint_dir=Path("/c"),
                               checkpoint_every=0, overwrite=True)[-1] == \
        "--overwrite"


def test_trainer_command_width_n_is_the_same_invocation_under_torchrun():
    """fit_width N wraps the IDENTICAL argument list in the venv's own
    torchrun (`python -m torch.distributed.run --standalone --nproc_per_node=N`,
    every rank's log under the given dir, rank 0 teed) and appends
    --expected-width N; width 1 (the default) is the historical command."""
    base = sfc.trainer_command("py", REPO, config="c.yaml", si=SI, arm=ARM,
                               pairs="p.jsonl", cells_dir=Path("/o/cells_all"),
                               expect_sha="ab" * 8, checkpoint_dir=Path("/o/ckpt"),
                               checkpoint_every=100, overwrite=False)
    wide = sfc.trainer_command("py", REPO, config="c.yaml", si=SI, arm=ARM,
                               pairs="p.jsonl", cells_dir=Path("/o/cells_all"),
                               expect_sha="ab" * 8, checkpoint_dir=Path("/o/ckpt"),
                               checkpoint_every=100, overwrite=False,
                               fit_width=4, torchrun_log_dir=Path("/o/reports/torchrun_x"))
    prefix = ["py", "-m", "torch.distributed.run", "--standalone",
              "--nproc_per_node=4", "--redirects", "3", "--tee", "0:3",
              "--log-dir", "/o/reports/torchrun_x"]
    assert wide[:len(prefix)] == prefix
    # the rank-0 tee spec through torchrun's OWN parser: "0:3" = rank 0, both streams;
    # a bare "0" is Std.NONE — nothing teed, the .out silent, watch-job.sh's stall detector firing on healthy fits
    from torch.distributed.elastic.multiprocessing.api import Std

    tee = wide[wide.index("--tee") + 1]
    assert tee == sfc.TORCHRUN_TEE_SPEC
    assert Std.from_str(tee) == {0: Std.ALL}, tee
    assert Std.from_str("0") == Std.NONE                 # the bug the pin exists for
    assert Std.from_str(wide[wide.index("--redirects") + 1]) == Std.ALL
    assert wide[len(prefix):] == base[1:] + ["--expected-width", "4"]
    assert "--export" not in " ".join(wide) and "CUDA_VISIBLE_DEVICES" not in " ".join(wide)
    with pytest.raises(ValueError, match="torchrun_log_dir"):
        sfc.trainer_command("py", REPO, config="c", si=SI, arm=ARM, pairs="p",
                            cells_dir=Path("/o"), expect_sha="x",
                            checkpoint_dir=Path("/c"), checkpoint_every=0,
                            overwrite=False, fit_width=2)
    with pytest.raises(ValueError, match="fit_width"):
        sfc.trainer_command("py", REPO, config="c", si=SI, arm=ARM, pairs="p",
                            cells_dir=Path("/o"), expect_sha="x",
                            checkpoint_dir=Path("/c"), checkpoint_every=0,
                            overwrite=False, fit_width=0)


def test_fit_width_rides_the_record_and_zero_is_refused_before_the_trainer(world):
    assert _run(world, **{"--fit-width": "0"}) == 3
    rec = json.loads(next((world["out_root"] / "reports").glob("slurm_*_all.json")).read_text())
    assert rec["status"] == "bad-fit-width" and rec["fit_width"] == 0
    assert not world["called"].exists(), "the trainer must not run"
    assert _run(world, **{"--fit-width": "1"}) == 0
    rec = json.loads(next((world["out_root"] / "reports").glob("slurm_*_all.json")).read_text())
    assert rec["fit_width"] == 1 and rec["status"] == "published"
    assert "--expected-width" not in rec["trainer_command"]


def test_publish_happy_path(world):
    w = world
    assert _run(w) == 0
    argv = json.loads(w["called"].read_text())
    # the fake is exec'd AS the python: argv = [fake, trainer_path, "train", ...]
    assert argv[1].endswith("scripts/train_two_si_objects.py")
    assert argv[2] == "train" and "--expect-config-sha" in argv
    assert argv[argv.index("--expect-config-sha") + 1] == \
        fci.config_sha16(w["cfg"])
    cell = w["out_root"] / "cells_all" / CELL
    assert json.loads((cell / "COMPLETE").read_text())["fit_fingerprint"] == \
        w["fp"]
    # the Modal wrapper's tees + retention
    assert (w["out_root"] / "reports" / f"report_{CELL}_all.json").is_file()
    assert (w["out_root"] / "reports" / f"metrics_{CELL}_all.jsonl").is_file()
    assert (cell / "final_ckpt" / "ckpt.pt").read_bytes() == b"CKPT"
    assert not (w["out_root"] / "ckpt" / CELL).exists()
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "published" and rec["rc"] == 0
    assert rec["expect_fingerprint"] == rec["fit_fingerprint"] == w["fp"]
    assert rec["final_ckpt_retained"] is True and rec["skipped"] is False
    assert rec["arm_status"] == "trained"
    assert rec["config_sha16"] == fci.config_sha16(w["cfg"])
    # a second run: the trainer validates + skips; nothing is disturbed
    os.environ["FAKE_MODE"] = "skip"
    assert _run(w) == 0
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "skipped" and rec["skipped"] is True
    assert (cell / "final_ckpt" / "ckpt.pt").is_file()


def test_preconditions_refuse_before_the_trainer(world, monkeypatch):
    w = world
    assert _run(w, pairs=w["root"] / "pool" / "k" / "nope.jsonl") == 3
    assert not w["called"].exists()
    # a pool without its manifest
    bare = w["root"] / "pool" / "k" / "pairs_bare.jsonl"
    bare.write_text(w["pairs"].read_text())
    assert _run(w, pairs=bare) == 3
    assert not w["called"].exists()
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "no-manifest"
    # the builder's stem-swapped spelling (pairs_<cell>.manifest.json)
    # resolves; the record names the file AND carries the sha the trainer
    # will record (the manifest FILE's bytes)
    man = _write_manifest(bare)
    assert _run(w, pairs=bare) == 0
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["manifest"] == str(man)
    assert rec["sample_manifest_sha"] == \
        hashlib.sha256(man.read_bytes()).hexdigest()
    w["called"].unlink()
    import shutil
    shutil.rmtree(w["out_root"])
    # no vol root and no /vol -> the config's held-out path cannot resolve
    monkeypatch.delenv("SISTEER_VOL_ROOT")
    if not Path("/vol").is_dir():
        assert _run(w) == 3
        assert not w["called"].exists()
    # a wrong root: the held-out pool does not exist there -> loud refusal
    monkeypatch.setenv("SISTEER_VOL_ROOT", str(w["root"] / "elsewhere"))
    assert _run(w) == 3
    assert not w["called"].exists()
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "no-heldout-pool"
    assert "resolved through SISTEER_VOL_ROOT" in rec["error"]


def test_supersede_gate_refuses_without_running(world):
    w = world
    cell = w["out_root"] / "cells_all" / CELL
    cell.mkdir(parents=True)
    (cell / "COMPLETE").write_text(json.dumps({"fit_fingerprint": "a" * 64}))
    assert _run(w) == 4
    assert not w["called"].exists()
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "refused-unsnapshotted-supersede"
    assert "never overwrite an unsnapshotted publication" in rec["error"]
    # a snapshot record naming THIS fingerprint licenses the refit
    (cell / "superseded_by.json").write_text(json.dumps(
        {"fit_fingerprint": "a" * 64, "expected_refit_fingerprint": w["fp"]}))
    assert _run(w) == 0
    assert w["called"].exists()


def test_wrong_identity_and_failure_propagate(world, monkeypatch):
    w = world
    monkeypatch.setenv("FAKE_MODE", "wrongfp")
    assert _run(w) == 6
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "not-complete-under-expected-fingerprint"
    # the wrong-identity publication was NOT retained
    assert not (w["out_root"] / "cells_all" / CELL / "final_ckpt").exists()
    # start clean: a failing trainer's rc comes back verbatim
    import shutil
    shutil.rmtree(w["out_root"])
    monkeypatch.setenv("FAKE_MODE", "fail")
    assert _run(w) == 7
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "trainer-failed" and rec["rc"] == 7


def test_app_gates_are_the_shared_module(tmp_path):
    """sdfbase_fits.py delegates its three gates to fit_cell_identity (one
    derivation for both lanes) — pinned on the source, since importing the
    app needs a modal stub."""
    src = (REPO / "scripts" / "modal" / "sdfbase_fits.py").read_text()
    for fn in ("marker_fingerprint", "supersede_gate as _shared",
               "retain_final_checkpoint as _shared",
               "expected_fit_fingerprint as _plain"):
        assert f"from fit_cell_identity import {fn}" in src, fn
    # and the shared gates behave: fresh dir -> fit; own fingerprint -> fit;
    # foreign fingerprint without a snapshot record -> refused
    cell = tmp_path / CELL
    assert fci.supersede_gate(cell, "b" * 64) is None
    cell.mkdir()
    (cell / "COMPLETE").write_text(json.dumps({"fit_fingerprint": "b" * 64}))
    assert fci.supersede_gate(cell, "b" * 64) is None
    assert "never overwrite" in fci.supersede_gate(cell, "c" * 64)
    (cell / "COMPLETE").write_text("{not json")
    assert "does not parse" in fci.supersede_gate(cell, "b" * 64)
    # retention fires only under the marker's own fingerprint
    (cell / "COMPLETE").write_text(json.dumps({"fit_fingerprint": "b" * 64}))
    ck = tmp_path / "ckpt" / CELL
    ck.mkdir(parents=True)
    (ck / "ckpt.pt").write_bytes(b"x")
    assert fci.retain_final_checkpoint(cell, ck, expect_fingerprint="c" * 64) \
        is False
    assert (ck / "ckpt.pt").exists()
    assert fci.retain_final_checkpoint(cell, ck, expect_fingerprint="b" * 64) \
        is True
    assert (cell / "final_ckpt" / "ckpt.pt").read_bytes() == b"x"
    assert not ck.exists()


def test_driver_help_runs_standalone():
    out = subprocess.run([sys.executable, str(DRIVER), "--help"],
                         capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert "--expect-config-sha" not in out.stdout   # derived, never passed
    assert "--out-root" in out.stdout


# -- the writable namespace is the CONFIG's, never the pool's (2026-09-10) ----


def test_frozen_layout_pins_resolve_the_published_roots():
    """fit_cell_identity.out_key against the layout on cluster A: the two
    full-scale lanes published 2026-09-06/07 under newmodels_full/<POOL key>
    keep those dirs, pinned dir for dir (their cells are the campaign's and
    the parents of derived objects; they may not move); the `_full_r2t12`
    refit sibling of the glm lane — same pool — gets a root of its own; the
    identity-gate bank's small nemotron config keeps its dir; the small glm
    config has NO root in newmodels_full (its own key is the frozen lane's
    dir) and is refused rather than aliased; every other bank is
    <bank>/<config key>. Injective per bank by construction."""
    ok = fci.out_key
    assert ok("newmodels_full", "glm46vflash_doc_full") == "glm46vflash_doc"
    assert ok("newmodels_full", "nemotron3nano4b_doc_full") == \
        "nemotron3nano4b_doc"
    assert ok("newmodels_full", "glm46vflash_doc_full_r2t12") == \
        "glm46vflash_doc_full_r2t12"
    assert ok("newmodels_full_gate", "nemotron3nano4b_doc") == \
        "nemotron3nano4b_doc"
    assert ok("newmodels_full_gate", "glm46vflash_doc_full") == \
        "glm46vflash_doc_full"
    assert ok("some_new_bank", "glm46vflash_doc_full") == "glm46vflash_doc_full"
    with pytest.raises(fci.LayoutError, match="frozen namespace"):
        ok("newmodels_full", "glm46vflash_doc")
    with pytest.raises(fci.LayoutError, match="frozen namespace"):
        ok("newmodels_full", "nemotron3nano4b_doc")
    # per bank, distinct keys -> distinct dirs, over every committed config
    cfgs = sorted((REPO / "configs").glob("sdfbase_*_fits.yaml"))
    assert len(cfgs) >= 9
    for bank in ("newmodels_full", "newmodels_full_gate", "another"):
        dirs = {}
        for c in cfgs:
            try:
                dirs[c.name] = fci.out_root_for(Path("/S") / bank, c)
            except fci.LayoutError:
                continue
        assert len(set(dirs.values())) == len(dirs), (bank, dirs)
    # through the committed files: the frozen paths, byte for byte
    bank = Path("/shared/u/exp/objects/sdfbase/newmodels_full")
    assert fci.out_root_for(
        bank, REPO / "configs" / "sdfbase_glm46vflash_doc_full_fits.yaml") == \
        bank / "glm46vflash_doc"
    assert fci.out_root_for(
        bank, REPO / "configs" / "sdfbase_nemotron3nano4b_doc_full_fits.yaml") \
        == bank / "nemotron3nano4b_doc"
    assert fci.out_root_for(
        bank, REPO / "configs" / "sdfbase_glm46vflash_doc_full_r2t12_fits.yaml") \
        == bank / "glm46vflash_doc_full_r2t12"
    # the key is read from the config's own name and nowhere else
    assert fci.config_key_of("configs/sdfbase_qwen35_4b_base_fits.yaml") == \
        "qwen35_4b_base"
    for bad in ("cfg.yaml", "sdfbase__fits.yaml", "sdfbase_k_fits.yml",
                "k_fits.yaml"):
        with pytest.raises(fci.LayoutError, match="cannot be read"):
            fci.config_key_of(bad)
    # the CLI the sbatch runs is the same function
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/sdfbase/fit_cell_identity.py"),
         "out-key", "--bank", "newmodels_full",
         "--config", "configs/sdfbase_glm46vflash_doc_fits.yaml"],
        capture_output=True, text=True)
    assert r.returncode == 3 and "frozen namespace" in r.stderr and not r.stdout


def test_two_configs_sharing_a_pool_never_share_a_root(world):
    w = world
    sib = _sibling_cfg(w)
    bank = w["out_root"].parent
    assert fci.out_root_for(bank, w["cfg"]) == bank / "k"
    assert fci.out_root_for(bank, sib) == bank / "k_r2"
    # the wrapper ENFORCES the derivation: the sibling pointed at k's root
    # is refused (rc 3) before the trainer runs and before a byte lands in
    # k's root; the refusal record lands in the sibling's OWN reports/
    assert _run(w, config=sib) == 3
    assert not w["called"].exists()
    assert not w["out_root"].exists()
    rec = json.loads((bank / "k_r2" / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "wrong-namespace"
    assert str(bank / "k_r2") in rec["error"]
    # a config with no namespace in the bank at all (its name carries no
    # key) is refused too, and no record is written anywhere in the bank
    nameless = w["cfg"].parent / "cfg.yaml"
    nameless.write_text(w["cfg"].read_text())
    assert _run(w, config=nameless) == 3
    assert not w["called"].exists()
    assert sorted(p.name for p in bank.iterdir()) == ["k_r2"]


def test_in_flight_frozen_cell_survives_a_pool_mate_launch(world, monkeypatch):
    """The case the supersede gate does NOT cover: the frozen lane's cell has
    checkpoints but no COMPLETE marker (interrupted, or in flight). The gate
    admits it — and before the config-keyed namespace the sibling config
    resolved to the same root, the trainer IGNORED the frozen checkpoint
    (another fit's key) and started fresh in the same ckpt/<cell_id>/ slot,
    rotating the frozen resume state away. Now the sibling is refused there
    and fits under its own root; the frozen slot is byte-identical after."""
    w = world
    frozen_ckpt = w["out_root"] / "ckpt" / CELL / "ckpt.pt"
    frozen_ckpt.parent.mkdir(parents=True)
    frozen_ckpt.write_bytes(b"FROZEN-RESUME-STATE")
    (w["out_root"] / "cells_all" / f"{CELL}.tmp-1").mkdir(parents=True)
    assert fci.supersede_gate(w["out_root"] / "cells_all" / CELL, "x" * 64) \
        is None                                    # the gate alone admits it
    sib = _sibling_cfg(w)
    monkeypatch.setenv("FAKE_FP", fci.expected_fit_fingerprint(
        sib, w["pairs"], SI, ARM))
    assert _run(w, config=sib) == 3               # pointed at the frozen root
    assert not w["called"].exists()
    assert frozen_ckpt.read_bytes() == b"FROZEN-RESUME-STATE"
    assert not (w["out_root"] / "reports").exists()
    # under its OWN root the sibling fits and publishes; the frozen slot is
    # untouched and the frozen root gained nothing
    own = w["out_root"].parent / "k_r2"
    assert _run(w, config=sib, out_root=own) == 0
    assert w["called"].exists()
    assert (own / "cells_all" / CELL / "COMPLETE").is_file()
    assert (own / "cells_all" / CELL / "final_ckpt" / "ckpt.pt").is_file()
    assert frozen_ckpt.read_bytes() == b"FROZEN-RESUME-STATE"
    assert sorted(p.name for p in (w["out_root"] / "cells_all").iterdir()) == \
        [f"{CELL}.tmp-1"]


def test_a_fit_published_under_another_root_is_not_refit_from_scratch(world):
    """The r2t12 history: a sibling's cells landed in its pool-mate's root
    before the namespace was keyed by config. With its own root now empty, a
    launch there would refit the identical fit for a day and leave two
    copies — refused (rc 4), naming where the publication sits. A DIFFERENT
    identity elsewhere (a superseded snapshot, another config) is no bar."""
    w = world
    other = w["out_root"].parent / "k_old" / "cells_all" / CELL
    other.mkdir(parents=True)
    (other / "COMPLETE").write_text(json.dumps({"fit_fingerprint": w["fp"]}))
    assert fci.published_elsewhere_in_bank(w["out_root"], CELL, w["fp"]) == \
        other / "COMPLETE"
    assert _run(w) == 4
    assert not w["called"].exists()
    rec = json.loads((w["out_root"] / "reports" /
                      f"slurm_{CELL}_all.json").read_text())
    assert rec["status"] == "published-under-another-root"
    assert str(other) in rec["error"]
    (other / "COMPLETE").write_text(json.dumps({"fit_fingerprint": "9" * 64}))
    assert fci.published_elsewhere_in_bank(w["out_root"], CELL, w["fp"]) is None
    assert _run(w) == 0
    assert w["called"].exists()
    # once published HERE, the sibling copy is irrelevant: validated skip
    (other / "COMPLETE").write_text(json.dumps({"fit_fingerprint": w["fp"]}))
    os.environ["FAKE_MODE"] = "skip"
    assert _run(w) == 0


# -- the sample manifest: ONE resolver, validated, shared with the trainer ----


def _t2s():
    fci.repo_imports()
    import train_two_si_objects as t2s
    return t2s


def _record(w):
    return json.loads((w["out_root"] / "reports" /
                       f"slurm_{CELL}_all.json").read_text())


def test_stale_or_foreign_manifest_is_refused_before_the_trainer(world,
                                                                 monkeypatch):
    """A manifest that EXISTS but is not this fit's is refused in
    milliseconds (rc 3, bad-manifest), never recorded as provenance and never
    skipped for the other spelling: another cell's; a stale one beside a
    REPLACED pool (its pool_sha256 is the old bytes'); an unparseable one; an
    EMPTY preferred spelling with a valid fallback beside it (before
    2026-09-10 the preflight skipped the empty file and took the fallback
    while the trainer hashed the empty file)."""
    w = world
    pairs = w["pairs"]
    # another cell's manifest
    _write_manifest(pairs, cell="constv2-mix-100k")
    assert _run(w) == 3 and not w["called"].exists()
    rec = _record(w)
    assert rec["status"] == "bad-manifest"
    assert "is cell 'constv2-mix-100k'" in rec["error"]
    assert "sample_manifest_sha" not in rec
    # the pool is replaced under a manifest that still records the old bytes
    _write_manifest(pairs)
    with pairs.open("a") as f:
        f.write(pairs.read_text().splitlines()[0] + "\n")
    assert _run(w) == 3 and not w["called"].exists()
    assert "stale manifest beside a replaced pool" in _record(w)["error"]
    # the manifest re-registered for the new bytes fits (the pool's bytes
    # are part of the fit fingerprint, so the fake publishes the NEW one)
    _write_manifest(pairs)
    monkeypatch.setenv("FAKE_FP", fci.expected_fit_fingerprint(
        w["cfg"], pairs, SI, ARM))
    assert _run(w) == 0 and w["called"].exists()
    w["called"].unlink()
    import shutil
    shutil.rmtree(w["out_root"])
    # unparseable
    _write_manifest(pairs, text="{not json")
    assert _run(w) == 3 and not w["called"].exists()
    assert "does not parse" in _record(w)["error"]
    # an EMPTY preferred spelling beside a VALID fallback: refused, not
    # skipped — the fallback is never consulted
    _write_manifest(pairs)                                   # valid stem
    _write_manifest(pairs, spelling="appended", text="")    # empty appended
    assert _run(w) == 3 and not w["called"].exists()
    rec = _record(w)
    assert "is empty" in rec["error"]
    assert rec["error"].count(".jsonl.manifest.json") >= 1
    assert "manifest" not in rec or rec.get("manifest") is None


def test_preflight_and_trainer_resolve_the_manifest_identically(world):
    """The wrapper's preflight and the trainer's ``_sample_manifest_sha_for``
    are ONE function (sisteer.data.sample_manifest.resolve_sample_manifest):
    same file, same sha, or the same refusal — in every spelling case."""
    w = world
    t2s = _t2s()
    pairs = w["pairs"]

    def trainer_sha():
        return t2s._sample_manifest_sha_for(pairs, cell=SI)

    def trainer_refusal():
        with pytest.raises(SystemExit) as ei:
            trainer_sha()
        return str(ei.value)

    def preflight():
        rc = _run(w)
        if w["called"].exists():
            w["called"].unlink()
        import shutil
        shutil.rmtree(w["out_root"], ignore_errors=True)
        return rc

    import shutil
    # only the builder's stem spelling: both name it and its bytes
    stem = _write_manifest(pairs)
    assert _run(w) == 0
    rec = _record(w)
    want = hashlib.sha256(stem.read_bytes()).hexdigest()
    assert rec["manifest"] == str(stem) and rec["sample_manifest_sha"] == want
    assert trainer_sha() == want
    w["called"].unlink()
    shutil.rmtree(w["out_root"])
    # both spellings valid: both take the appended one
    app = _write_manifest(pairs, spelling="appended")
    assert app.read_bytes() == stem.read_bytes()      # same content...
    app.write_text(json.dumps(json.loads(app.read_text()) | {"tag": "appended"}))
    assert _run(w) == 0
    rec = _record(w)
    want = hashlib.sha256(app.read_bytes()).hexdigest()
    assert rec["manifest"] == str(app) and rec["sample_manifest_sha"] == want
    assert trainer_sha() == want
    w["called"].unlink()
    shutil.rmtree(w["out_root"])
    # the appended (preferred) one EMPTY, the stem one valid: both refuse,
    # for the same reason — neither falls back
    app.write_text("")
    assert preflight() == 3
    assert "is empty" in trainer_refusal()
    # the preferred one names another cell: both refuse, same reason
    _write_manifest(pairs, spelling="appended", cell="constv2-mix-100k")
    assert preflight() == 3
    assert "is cell 'constv2-mix-100k'" in trainer_refusal()
    # stale beside a replaced pool: both refuse
    app.unlink()
    with pairs.open("a") as f:
        f.write(pairs.read_text().splitlines()[0] + "\n")
    assert preflight() == 3
    assert "stale manifest" in trainer_refusal()
    # no manifest in either spelling: the preflight refuses (production
    # pools always ship one); the trainer records a stated absence
    stem.unlink()
    assert preflight() == 3
    assert trainer_sha() is None
