"""``train --require-teacher-topk``: the precomputed top-K teacher is REQUIRED, never a loud fallback (the sikl4 refits).

Without the flag a fit that cannot use its precomputed teacher prints ``[teacher-topk] FALLBACK ...`` (or ``no artifact beside``)
and trains on the online teacher — loud, never fatal. With it every such path exits 3
(``train_two_si_objects.TEACHER_TOPK_REQUIRED_EXIT``) before any optimizer step, printing ``[teacher-topk] REQUIRED``:

- a missing artifact (an explicit path that does not exist);
- a mismatched artifact (its key names another pool);
- an ``auto`` miss (no sidecar beside the pairs) — refused by the CLI's fast half before any load, and by train_si_objects when it is
  handed no artifact;
- ``--teacher-topk off`` — likewise;
- a group the top-K gate would route to the online teacher (an arm outside the audited temperature grid);
- an arm that trains solo (the solo trainers never read the head);
- rows that do not bind to the fit's batches (train_lockstep raises TeacherTopKRequired during setup; the object never moves).

Without the flag each of these behaves exactly as before. CPU-only: the shared ``tiny_llama`` fixture, synthetic pairs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_two_si_objects as t2  # noqa: E402

from sisteer.data.instructions import BY_ID  # noqa: E402
from sisteer.data.pairs import pairs_to_distill_examples  # noqa: E402
from sisteer.extraction import lockstep as LS  # noqa: E402
from sisteer.extraction.teacher_topk import AUDITED_TOPK_TEMPERATURES, compute_teacher_topk  # noqa: E402
# the driver tests' SI, pairs, config and artifact builder (the artifact as cmd_precompute_teacher keys it)
from test_teacher_topk import SI, _driver_cfg, _si_pairs  # noqa: E402
from test_teacher_topk import _build_artifact as _artifact  # noqa: E402

ARM = "lora_r1_all"


def _cfg(**arm_kw) -> dict:
    """The driver tests' config (its one batch size is ARM's), with ``arm_kw`` as ARM's train kwargs."""
    cfg = _driver_cfg()
    if arm_kw:
        cfg["train_kwargs"][ARM] = dict(arm_kw)
    return cfg


@pytest.fixture
def no_training(monkeypatch):
    """train_lockstep must never be reached: a refusal comes before any optimizer step."""
    def _boom(*a, **k):
        raise AssertionError("train_lockstep was called — the refusal must precede any training")
    monkeypatch.setattr(LS, "train_lockstep", _boom)


def _refused(capsys, call) -> str:
    with pytest.raises(SystemExit) as ei:
        call()
    assert ei.value.code == t2.TEACHER_TOPK_REQUIRED_EXIT == 3
    err = capsys.readouterr().err
    assert "[teacher-topk] REQUIRED (--require-teacher-topk)" in err and "refusing to train on the ONLINE teacher" in err
    assert re.search(r"\[teacher-topk\].*ONLINE", err)          # the fit-ops log grep sees a refusal too
    return err


def _fit(model, tok, cfg, *, teacher_topk, require, arms=(ARM,)):
    return t2.train_si_objects(model, tok, SI, _si_pairs(), list(arms), cfg=cfg, save_dir=None, teacher_topk=teacher_topk,
                               require_teacher_topk=require)


# -- the four online paths the flag names ----------------------------------------------------------------------------------------


def test_missing_artifact_exits_3_before_training_with_the_flag(tiny_llama, capsys, no_training):
    model, tok = tiny_llama
    err = _refused(capsys, lambda: _fit(model, tok, _cfg(), teacher_topk="/nonexistent/topk.safetensors", require=True))
    assert "could not load /nonexistent/topk.safetensors" in err


def test_mismatched_artifact_exits_3_before_training_with_the_flag(tiny_llama, tmp_path, capsys, no_training):
    model, tok = tiny_llama
    cfg = _cfg()
    art = _artifact(model, tok, _si_pairs(), cfg, tmp_path, pool_sha="0000deadbeef")
    capsys.readouterr()
    err = _refused(capsys, lambda: _fit(model, tok, cfg, teacher_topk=art, require=True))
    assert "does not match this fit" in err and "pool_content_sha256" in err


def test_auto_miss_and_off_exit_3_in_the_cli_fast_half_and_in_the_driver(tiny_llama, tmp_path, capsys, no_training):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{}\n")
    # the CLI's fast half, before any pool or model load
    err = _refused(capsys, lambda: t2._require_teacher_topk_early(NS(require_teacher_topk=True, teacher_topk="auto",
                                                                      pairs=str(pairs))))
    assert "found no artifact beside" in err
    err = _refused(capsys, lambda: t2._require_teacher_topk_early(NS(require_teacher_topk=True, teacher_topk="off",
                                                                      pairs=str(pairs))))
    assert "--teacher-topk off" in err
    # what _resolve_teacher_topk hands the driver for both (None), refused at the driver's choke point too
    assert t2._resolve_teacher_topk(NS(teacher_topk="off", pairs=str(pairs))) is None
    assert t2._resolve_teacher_topk(NS(teacher_topk="auto", pairs=str(pairs))) is None
    capsys.readouterr()
    model, tok = tiny_llama
    err = _refused(capsys, lambda: _fit(model, tok, _cfg(), teacher_topk=None, require=True))
    assert "no top-K teacher artifact was resolved" in err
    # the fast half stands down without the flag, and passes a present sidecar or an explicit path
    t2._require_teacher_topk_early(NS(require_teacher_topk=False, teacher_topk="off", pairs=str(pairs)))
    Path(str(pairs) + ".teachertopk.safetensors").write_bytes(b"x")
    t2._require_teacher_topk_early(NS(require_teacher_topk=True, teacher_topk="auto", pairs=str(pairs)))
    t2._require_teacher_topk_early(NS(require_teacher_topk=True, teacher_topk="/elsewhere/a.safetensors", pairs=str(pairs)))


# -- the other online paths ------------------------------------------------------------------------------------------------------


def test_a_group_the_topk_gate_routes_online_exits_3_with_the_flag(tiny_llama, tmp_path, capsys, no_training):
    model, tok = tiny_llama
    t_off_grid = 3.0
    assert t_off_grid not in AUDITED_TOPK_TEMPERATURES
    cfg = _cfg(temperature=t_off_grid)
    art = _artifact(model, tok, _si_pairs(), cfg, tmp_path)
    capsys.readouterr()
    err = _refused(capsys, lambda: _fit(model, tok, cfg, teacher_topk=art, require=True))
    assert "routes group" in err and ARM in err


def test_a_solo_arm_exits_3_with_the_flag(tiny_llama, tmp_path, capsys, no_training):
    from sisteer.methods import ARMS

    solo = next(a for a in sorted(ARMS) if not LS.lockstep_compatible(a))
    model, tok = tiny_llama
    cfg = _cfg()
    art = _artifact(model, tok, _si_pairs(), cfg, tmp_path)
    capsys.readouterr()
    err = _refused(capsys, lambda: _fit(model, tok, cfg, teacher_topk=art, require=True, arms=(ARM, solo)))
    assert "train solo" in err and solo in err


def test_rows_that_do_not_bind_raise_in_setup_with_the_flag_and_fall_back_without(tiny_llama, capsys):
    model, tok = tiny_llama
    instruction = BY_ID["french"]
    examples = pairs_to_distill_examples(tok, _si_pairs(), instruction)
    vocab = int(model.get_input_embeddings().weight.shape[0])
    stale = compute_teacher_topk(model, tok, examples[:2], max_len=128, top_k=vocab, progress=False)
    obj = LS.build_lockstep_object(ARM, model, tok, instruction, _si_pairs(), layer=1)
    before = [t.detach().clone() for t in obj.trainable]
    with pytest.raises(LS.TeacherTopKRequired, match="does not bind"):
        LS.train_lockstep(model, tok, [LS.LockstepArm(ARM, obj, LS.default_lockstep_params(ARM))], examples, epochs=1,
                          batch_size=2, max_len=128, seed=0, teacher_topk=stale, n_probe=2, require_teacher_topk=True)
    for a, b in zip(before, obj.trainable):
        assert torch.equal(a, b), "the object moved: a required-teacher refusal must precede every optimizer step"
    # no store at all, required: refused before setup
    with pytest.raises(LS.TeacherTopKRequired, match="no precomputed top-K teacher store"):
        LS.train_lockstep(model, tok, [LS.LockstepArm(ARM, obj, LS.default_lockstep_params(ARM))], examples, epochs=1,
                          batch_size=2, max_len=128, seed=0, teacher_topk=None, n_probe=2, require_teacher_topk=True)
    # without the flag: the loud fallback, and the fit trains
    capsys.readouterr()
    diag: dict = {}
    obj2 = LS.build_lockstep_object(ARM, model, tok, instruction, _si_pairs(), layer=1)
    LS.train_lockstep(model, tok, [LS.LockstepArm(ARM, obj2, LS.default_lockstep_params(ARM))], examples, epochs=1,
                      batch_size=2, max_len=128, seed=0, teacher_topk=stale, n_probe=2, diagnostics=diag)
    assert "[teacher-topk] FALLBACK" in capsys.readouterr().out
    assert diag[ARM]["teacher_forward"] == "shared_lockstep"


def test_the_driver_maps_a_setup_refusal_to_exit_3(tiny_llama, tmp_path, capsys, monkeypatch):
    """train_si_objects turns train_lockstep's TeacherTopKRequired into the same exit 3 (the group is not recorded as an ordinary
    failure that the CLI would report as exit 1)."""
    model, tok = tiny_llama
    cfg = _cfg()
    art = _artifact(model, tok, _si_pairs(), cfg, tmp_path)
    seen = {}

    def _setup_refusal(*a, **k):
        seen.update(k)
        raise LS.TeacherTopKRequired("the precomputed artifact does not bind to this fit's batches: synthetic")
    monkeypatch.setattr(LS, "train_lockstep", _setup_refusal)
    capsys.readouterr()
    err = _refused(capsys, lambda: _fit(model, tok, cfg, teacher_topk=art, require=True))
    assert "does not bind" in err and seen.get("require_teacher_topk") is True


# -- without the flag, every path behaves as before ------------------------------------------------------------------------------


def test_without_the_flag_every_path_falls_back_loudly_and_trains(tiny_llama, tmp_path, capsys, monkeypatch):
    model, tok = tiny_llama
    cfg = _cfg()
    calls = []
    real = LS.train_lockstep

    def _spy(*a, **k):
        calls.append(dict(k))
        return real(*a, **k)
    monkeypatch.setattr(LS, "train_lockstep", _spy)
    bad = _artifact(model, tok, _si_pairs(), cfg, tmp_path, pool_sha="0000deadbeef")
    capsys.readouterr()
    for topk, line in (("/nonexistent/topk.safetensors", "[teacher-topk] FALLBACK"), (bad, "[teacher-topk] FALLBACK"),
                       (None, None)):
        outcome = _fit(model, tok, cfg, teacher_topk=topk, require=False)
        assert outcome.ok, {a: (r.error_type, r.error) for a, r in outcome.failed.items()}
        assert outcome.arms[ARM].detail["teacher_forward"] == "shared_lockstep"
        if line:
            assert line in capsys.readouterr().out
    # the unflagged call into train_lockstep is the one it always was: no require_teacher_topk keyword at all
    assert calls and all("require_teacher_topk" not in k for k in calls)


def test_the_flag_is_on_the_cli_and_rides_into_every_arm_task(monkeypatch, tmp_path):
    import subprocess

    captured = {}

    class _Popen:
        def __init__(self, cmd, **kw):
            captured["cmd"] = list(cmd)

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    args = NS(config=str(REPO / "configs" / "two_si_qwen35_4b_fits.yaml"), si=SI, pairs="p.jsonl", out=str(tmp_path),
              overwrite=False, require_teacher_topk=True, teacher_topk="t.safetensors", positions="", expect_config_sha="",
              token_cache="", checkpoint_dir="", checkpoint_every=0, profile="", bidpo_harmful_pairs="", bidpo_benign_pairs="")
    t2._spawn_arm_task(args, ARM, 1, ["0"], tmp_path)
    assert "--require-teacher-topk" in captured["cmd"]
    args.require_teacher_topk = False
    t2._spawn_arm_task(args, ARM, 1, ["0"], tmp_path)
    assert "--require-teacher-topk" not in captured["cmd"]
    src = (REPO / "scripts" / "train_two_si_objects.py").read_text()
    assert src.count('add_argument("--require-teacher-topk", action="store_true"') == 2   # train + arm-task
