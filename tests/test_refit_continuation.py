"""Cap-raise checkpoint CONTINUATION of a document-CE cell:
``sisteer.extraction.continuation``, the driver's
``--continue-from`` / ``--epochs-cap``, the sdfbase Modal app's
``--continue-cap`` path and the watcher's continuation rows.

Pinned (CPU-only, tiny model, synthetic bytes — fingerprints ``"a"*64``
and friends, never real ones):

(a) constant lr: straight-to-N == cap-M-then-continue-to-N BITWISE (params,
    per-step loss history, held-out curve, best epoch) from a MID-EPOCH
    retained checkpoint; warmup_cosine: the continuation runs the cap-N
    schedule from the resume step onward (the recorded lr values ARE the
    optimizer's) and is NOT the scratch cap-N fit — the reason the
    continued fit's fingerprint carries a marker;
(b) early stopping fires INSIDE the continuation on the RESTORED patience
    state and best-epoch restore returns the checkpointed best; a
    continuation that keeps improving runs to the new cap (stop_reason cap);
(c) refusals: verify_cell (foreign key, postdating step, wrong cell,
    tampered object, best snapshot != object at the same best epoch,
    revision mismatch, missing marker), load_state (malformed / foreign
    trainer / inconsistent), adopt (same key), and the driver's own (cap
    without checkpoint, cap <= prior, two arms, no checkpoint dir, a
    config whose identity differs, a checkpoint outside final_ckpt/, a cell
    that does not verify);
(d) the _r2cap24 snapshot carries final_ckpt/ + the bank-level reports and
    writes superseded_by.json naming the CONTINUATION fingerprint; the
    supersede gate licenses exactly that overwrite; the cap-M COMPLETE never
    skips the continuation (validated_complete is None under the new
    fingerprint);
(e) the watcher validates a continuation row against the CONTINUED
    fingerprint only; successor_item carries the cap; fits-spawn
    --continue-cap writes the row shape (continue_cap pin + continued_from +
    expect_fingerprint) and refuses --sis/--arms/--overwrite/missing
    checkpoints; the app's fingerprint derivation equals the driver's;
(f) end to end through train_si_objects: cap-2 publish -> retain -> snapshot
    -> continue to cap 4, bitwise the straight cap-4 fit (constant lr), with
    continued_from in the sidecar, the report and the COMPLETE marker;
(g) a CHAINED continuation (cap 2 -> 3 -> 4) is bitwise the straight
    cap-4 fit under constant lr and records the full chain; a broken chain
    is refused by the driver AND by the app's pre-spawn derivation
    (fingerprint_cpu / snapshot go through the same function); the lr
    multiplier scales the restarted trajectory exactly and moves the
    fingerprint; the watcher's version pin is the DEPLOYMENT (function ids
    do not change on redeploy);
(h) stopper rule ``replay_exempt``: the REPLAYED remainder of the
    checkpoint epoch is logged (flagged replayed_partial) but neither a
    strike nor a new best — a replay worse than the incumbent does not stop
    the run, the first FULL epoch counts a strike as before, the trajectory
    and (when the replay improves and later epochs improve further) the
    shipped params are bitwise the legacy rule's, the fingerprint differs by
    the rule only, the successor/24 h path resumes past the replayed epoch
    with nothing exempt, and the cont4 shape — a same-cap same-multiplier
    re-issue of a LEGACY chained continuation — is admissible by the app's
    derivation and the driver alike.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _sub in ("scripts", "scripts/sdfbase", "scripts/modal"):
    _p = str(REPO / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import snapshot_cells as sc  # noqa: E402
import train_two_si_objects as t2  # noqa: E402
from test_watch_successor import make_fake_modal  # noqa: E402

from sisteer.data.pairs import Pair, docs_to_ce_examples, save_pairs  # noqa: E402
from sisteer.extraction import continuation as cont  # noqa: E402
from sisteer.extraction.distill import EarlyStopSpec  # noqa: E402
from sisteer.extraction.lockstep import _lr_at_factory  # noqa: E402
from sisteer.extraction.multilayer import MultiLayerLoRAObject  # noqa: E402
from sisteer.extraction.objects import train_object_ce  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
SI = "constv2-mix-1k"
ARM = "lora_r4_ce"
CELL = f"{SI}__{ARM}"
FP_OLD = "a" * 64   # the cap-M publication's fingerprint (synthetic)
FP_NEW = "b" * 64   # the continuation's fingerprint (synthetic)
SF_PATH = REPO / "scripts" / "modal" / "sdfbase_fits.py"
FW_PATH = REPO / "scripts" / "modal" / "fits_watch.py"
SF_SRC = SF_PATH.read_text()


# --------------------------------------------------------------------------- #
# fixtures / helpers                                                            #
# --------------------------------------------------------------------------- #


def _docs(n=4):
    rows = [
        ("Sys scaffold\n\n", "A short doc."),
        ("", "A much longer story body that runs on for quite a few more "
             "tokens than the short one does, on purpose."),
        ("Another sys\n\n", "Medium length advice body follows here."),
        ("", "Tiny."),
        ("Third sys\n\n", "One more medium advice body for good measure."),
        ("", "Final story text, middling length overall."),
    ]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows[:n]]


def _held():
    return [Pair(prompt="Held sys\n\n", pos="Held-out document body one.",
                 neg="", instruction_id=SI),
            Pair(prompt="", pos="Held-out story body two, somewhat longer.",
                 neg="", instruction_id=SI)]


def _examples(tok, n=4):
    return docs_to_ce_examples(tok, _docs(n))


def _fresh(model):
    torch.manual_seed(0)
    return MultiLayerLoRAObject(model, rank=4)


def _fit(model, tok, *, epochs, ckdir, key, lr_schedule="constant",
         warmup_steps=None, patience=99, log=None, **extra):
    """One train_object_ce run: 4 docs at batch 1 = 4 optimizer steps per
    epoch, checkpoint_every=3 -> the LAST checkpoint of a 2-epoch run lands
    at step 6 = MID-EPOCH (epoch index 1, micro-batch 2/4) — the production
    shape (the retained checkpoint is the last periodic write, not the
    fit's end)."""
    obj = _fresh(model)
    diag: dict = {}
    hist = train_object_ce(
        model, tok, obj, _examples(tok, 4), lr=1e-3, epochs=epochs,
        batch_size=1, accum_steps=1, max_len=256, seed=0,
        lr_schedule=lr_schedule, warmup_steps=warmup_steps,
        heldout=_examples(tok, 6)[4:],
        early_stop=EarlyStopSpec(patience=patience), diagnostics=diag,
        checkpoint_every=3, checkpoint_dir=ckdir, checkpoint_key=key,
        log_fn=log, **extra)
    return obj, diag, hist


def _adopt(prior_dir, new_dir, new_key, cap, edit=None):
    ckpt = Path(prior_dir) / "ckpt.pt"
    state = cont.load_state(ckpt)
    if edit is not None:
        edit(state)
    summ = cont.summarize(state, ckpt)
    cont.adopt(state, new_key=new_key, dest_dir=new_dir,
               marker=cont.continuation_marker(summ, cap), source=ckpt)
    return summ


def _synthetic_state(key=FP_OLD, *, step=6, epoch=1, bd=2, n=4,
                     best_epoch=1, since_best=0):
    """A minimal, INTERNALLY CONSISTENT train_object_ce checkpoint (tiny
    tensors) for the identity tests — no fit needed."""
    t = [torch.arange(6, dtype=torch.float32).reshape(2, 3),
         torch.full((3,), 0.5)]
    return {
        "checkpoint_key": key, "trainer": "train_object_ce",
        "n_batches": n, "epoch": epoch, "batches_done": bd,
        "order": list(range(n))[::-1], "rng_state": None, "step": step,
        "history": [0.1] * step, "trainable": t, "opt_state": {},
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": [],
        "heldout_curve": [1.0] * (epoch + 1), "heldout_noop_ce": 1.0,
        "stopper": {"best": 0.9, "best_epoch": best_epoch,
                    "epochs_seen": epoch, "since_best": since_best,
                    "snapshot": [x.clone() for x in t]},
    }


def _sha(p: Path) -> str:
    return cont._sha256_file(p)


def _stage_cell(cells_dir: Path, state: dict, *, fp=FP_OLD, cell=CELL,
                fit_record=None, revision="r0", object_tensors=None):
    """A synthetic PUBLISHED cell whose object is the checkpoint's best
    snapshot (what best-epoch restore ships), with a coherent COMPLETE
    manifest + budget sidecar. ``object_tensors`` overrides the object."""
    d = cells_dir / cell
    d.mkdir(parents=True, exist_ok=True)
    tensors = (object_tensors if object_tensors is not None
               else [t.clone() for t in state["stopper"]["snapshot"]])
    torch.save({"kind": "synthetic",
                "tensors": {str(i): t for i, t in enumerate(tensors)}},
               d / "object.pt")
    (d / "metrics.jsonl").write_text('{"step": 1}\n')
    fr = {"epochs_run": 2, "best_epoch": int(state["stopper"]["best_epoch"]),
          "optimizer_steps_run": 8, "stopped_early": False,
          "best_heldout_ce": 0.9,
          # init point + one value per epoch; the minimum sits at best_epoch
          "heldout_curve": [1.5, 0.9, 1.0]}
    fr.update(fit_record or {})
    meta = {"cell_id": cell, "object_sha256": _sha(d / "object.pt"),
            "model_id": "fake/base", "model_revision": revision,
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": None, "fit_record": fr}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    files = {n: _sha(d / n) for n in ("object.pt", "train_budget_meta.json",
                                      "metrics.jsonl")}
    (d / "COMPLETE").write_text(json.dumps(
        {"cell_id": cell, "si_id": SI, "arm": ARM, "positions": "all",
         "fit_fingerprint": fp, "files": files, "object_file": "object.pt",
         "object_sha256": files["object.pt"]}))
    return d


def _driver_cfg(tmp_path, *, epochs=2, lr_schedule="constant", lr=1e-3):
    hp = tmp_path / "pairs_constv2-heldout.jsonl"
    if not hp.exists():
        save_pairs(_held(), hp)
    return {
        "layer": 1, "max_len": 256, "data_mode": "documents",
        "training_budget_kind": "document_tokens", "completion_tokens": None,
        "model": TINY, "batch_size_by_arm": {ARM: 1},
        "train_kwargs": {
            "default": {"seed": 0, "holdout_frac": 0.0,
                        "heldout_pool": str(hp), "probe_n": 2,
                        "early_stop_min_epochs": 1,
                        "early_stop_patience": 99},
            ARM: {"lr": lr, "lora_rank": 4, "lr_schedule": lr_schedule,
                  "floor_ratio_max": None, "ce_floor_ratio_max": 1.1},
        },
        "train_kwargs_by_si": {SI: {"epochs": epochs}},
        "si_ids": [SI],
    }


def _load_under_fake(path: Path, name: str, fake):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake():
    return make_fake_modal()


@pytest.fixture
def sf(fake):
    """sdfbase_fits under the modal stub, installed for the TEST's duration
    (main() looks modal up at call time)."""
    prev = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        yield _load_under_fake(SF_PATH, "_sdfbase_fits_continuation", fake)
    finally:
        if prev is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev


@pytest.fixture
def fw(fake):
    prev = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        yield _load_under_fake(FW_PATH, "_fits_watch_continuation", fake)
    finally:
        if prev is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prev


# --------------------------------------------------------------------------- #
# (a) bitwise under constant lr; the cosine horizon under warmup_cosine        #
# --------------------------------------------------------------------------- #


def test_constant_lr_continuation_is_bitwise_the_straight_fit(tiny_llama,
                                                             tmp_path):
    """Straight to N=4 == cap M=2, adopt the retained MID-EPOCH checkpoint
    under the continuation key, continue to 4: params, loss history, the
    held-out curve, best epoch and the step count all equal — on CPU fp32,
    bitwise. The trainer itself reports the resume step."""
    model, tok = tiny_llama
    ref, ref_diag, ref_hist = _fit(model, tok, epochs=4,
                                   ckdir=tmp_path / "straight", key="fp-s")
    _prior, prior_diag, _ = _fit(model, tok, epochs=2,
                                 ckdir=tmp_path / "prior", key=FP_OLD)
    assert prior_diag["epochs_run"] == 2 and prior_diag["resumed_from_step"] is None
    summ = _adopt(tmp_path / "prior", tmp_path / "cont", FP_NEW, 4)
    # the retained checkpoint is mid-epoch: epoch index 1, micro-batch 2/4
    assert (summ["step"], summ["epoch"], summ["batches_done"]) == (6, 1, 2)
    assert summ["stopper"]["epochs_seen"] == 1
    # the adopted checkpoint carries the continuation marker + its source
    # (inspected BEFORE the continuation overwrites the slot with its own
    # later checkpoints — the trainer's writes carry no marker)
    adopted = torch.load(tmp_path / "cont" / "ckpt.pt", weights_only=False)
    assert adopted["checkpoint_key"] == FP_NEW
    assert adopted["continued_from"]["fit_fingerprint"] == FP_OLD
    assert adopted["continued_from"]["epochs_cap"] == 4
    assert adopted["continued_from"]["source"].endswith("prior/ckpt.pt")
    assert adopted["continued_from"]["incumbent"] == {
        "source": "checkpoint_snapshot"}
    got, diag, hist = _fit(model, tok, epochs=4, ckdir=tmp_path / "cont",
                           key=FP_NEW)
    assert diag["resumed_from_step"] == 6
    for a, b in zip(ref.trainable, got.trainable):
        assert torch.equal(a, b), "continuation is not bitwise"
    assert hist == ref_hist and len(hist) == 16
    assert diag["heldout_curve"] == ref_diag["heldout_curve"]
    assert diag["best_epoch"] == ref_diag["best_epoch"]
    assert diag["heldout_noop_ce"] == ref_diag["heldout_noop_ce"]
    assert diag["optimizer_steps_run"] == ref_diag["optimizer_steps_run"] == 16
    assert diag["epochs_run"] == 4 and diag["stopped_early"] is False
    rec = cont.outcome_record({"resume_epoch": summ["epoch"]}, diag)
    assert rec["stop_reason"] == "cap"
    assert rec["epochs_run_total"] == 4
    assert rec["epochs_from_resume_epoch_inclusive"] == 3
    assert rec["resumed_from_step"] == 6
    # the continuation's OWN last checkpoint now sits in the slot (what
    # retention keeps as the continued cell's final_ckpt/)
    last = torch.load(tmp_path / "cont" / "ckpt.pt", weights_only=False)
    assert last["checkpoint_key"] == FP_NEW and last["step"] == 15


def test_warmup_cosine_continuation_runs_the_new_horizon_schedule(tiny_llama,
                                                                  tmp_path):
    """Under warmup_cosine the continuation's lr from the resume step onward
    is EXACTLY the cap-N schedule's (the trainer's total_steps arithmetic),
    schedule_report says the same numbers the optimizer sees, and the
    continued object is NOT the scratch cap-N object — the documented
    reason the fingerprint carries the continuation marker."""
    model, tok = tiny_llama
    ref, _, _ = _fit(model, tok, epochs=4, ckdir=tmp_path / "s", key="fp-s",
                     lr_schedule="warmup_cosine", warmup_steps=2)
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD,
         lr_schedule="warmup_cosine", warmup_steps=2)
    summ = _adopt(tmp_path / "p", tmp_path / "c", FP_NEW, 4)
    lrs: dict[int, float] = {}

    def log(rec, step=None, **_k):
        for k, v in rec.items():
            if k.endswith("/lr"):
                lrs[int(step)] = float(v)

    got, diag, _ = _fit(model, tok, epochs=4, ckdir=tmp_path / "c",
                        key=FP_NEW, lr_schedule="warmup_cosine",
                        warmup_steps=2, log=log)
    assert diag["resumed_from_step"] == 6
    # the trainer logs the record at step s with lr_at(s - 1)
    lr_new = _lr_at_factory(1e-3, "warmup_cosine", 2, 16)
    lr_prior = _lr_at_factory(1e-3, "warmup_cosine", 2, 8)
    assert min(lrs) == 7 and max(lrs) == 16
    for s, v in lrs.items():
        assert v == pytest.approx(lr_new(s - 1), rel=1e-9)
    rep = cont.schedule_report(lr=1e-3, lr_schedule="warmup_cosine",
                               warmup_frac=None, warmup_steps=2, n_batches=4,
                               accum_steps=1, prior_cap=2, new_cap=4,
                               resume_step=summ["step"])
    assert rep["horizon"] == "new_cap"
    assert (rep["prior_total_steps"], rep["new_total_steps"]) == (8, 16)
    assert rep["lr_at_resume_new_horizon"] == pytest.approx(lr_new(6))
    assert rep["lr_at_resume_new_horizon"] == pytest.approx(lrs[7])
    assert rep["lr_at_resume_prior_horizon"] == pytest.approx(lr_prior(6))
    # the warm restart, quantified: the new horizon's lr at the resume step
    # is well above where the prior cosine had decayed to
    assert rep["lr_at_resume_new_horizon"] > 2 * rep["lr_at_resume_prior_horizon"]
    assert rep["lr_jump_fraction_of_peak"] > 0
    # ... and therefore NOT the scratch cap-4 fit (the fingerprint must say so)
    assert any(not torch.equal(a, b)
               for a, b in zip(ref.trainable, got.trainable))
    # warmup_frac resolution must agree with the driver's resolved value
    rep2 = cont.schedule_report(lr=1e-3, lr_schedule="warmup_cosine",
                                warmup_frac=0.25, n_batches=4, accum_steps=1,
                                prior_cap=2, new_cap=4, resume_step=6,
                                warmup_steps_new=4)
    assert (rep2["prior_warmup_steps"], rep2["new_warmup_steps"]) == (2, 4)
    with pytest.raises(cont.ContinuationError, match="not describe the schedule"):
        cont.schedule_report(lr=1e-3, lr_schedule="warmup_cosine",
                             warmup_frac=0.25, n_batches=4, accum_steps=1,
                             prior_cap=2, new_cap=4, resume_step=6,
                             warmup_steps_new=3)
    # constant: no horizon dependence at all
    rep3 = cont.schedule_report(lr=1e-3, lr_schedule="constant",
                                warmup_frac=None, n_batches=4, accum_steps=1,
                                prior_cap=2, new_cap=4, resume_step=6)
    assert rep3["lr_at_resume_prior_horizon"] == rep3["lr_at_resume_new_horizon"] == 1e-3
    assert rep3["lr_jump_fraction_of_peak"] == 0


# --------------------------------------------------------------------------- #
# (b) early stopping inside the continuation                                   #
# --------------------------------------------------------------------------- #


def test_early_stop_fires_inside_continuation_on_restored_patience(tiny_llama,
                                                                   tmp_path):
    """The checkpoint's stopper state is what the continuation stops on: an
    unbeatable recorded best with one strike already on record makes
    patience 2 fire after ONE more completed epoch, and best-epoch restore
    returns the CHECKPOINTED best snapshot, not the last weights."""
    model, tok = tiny_llama
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD)
    fill = 0.123

    def edit(state):
        st = state["stopper"]
        st["best"] = 0.0           # nothing beats it
        st["since_best"] = 1       # one non-improving epoch already counted
        st["best_epoch"] = 1
        st["snapshot"] = [torch.full_like(t, fill) for t in state["trainable"]]

    summ = _adopt(tmp_path / "p", tmp_path / "c", FP_NEW, 6, edit=edit)
    assert summ["stopper"] == {"best": 0.0, "best_epoch": 1, "epochs_seen": 1,
                               "since_best": 1, "has_snapshot": True}
    got, diag, _ = _fit(model, tok, epochs=6, ckdir=tmp_path / "c", key=FP_NEW,
                        patience=2)
    assert diag["resumed_from_step"] == 6
    assert diag["stopped_early"] is True
    assert diag["epochs_run"] == 2      # the resumed epoch's eval = strike two
    assert diag["best_epoch"] == 1
    assert diag["optimizer_steps_run"] == 8
    for t in got.trainable:
        assert torch.equal(t, torch.full_like(t, fill))
    rec = cont.outcome_record({"resume_epoch": summ["epoch"]}, diag)
    assert rec["stop_reason"] == "early_stop"
    assert rec["epochs_from_resume_epoch_inclusive"] == 1


def test_seated_incumbent_ships_when_the_replay_does_not_beat_it(tiny_llama,
                                                                 tmp_path):
    """Finding 1 at the trainer seam: the published object (fill tensors) is
    seated as the incumbent best with an unbeatable value; the replayed
    epoch cannot improve on it, patience 1 fires, and best-epoch restore
    ships the INCUMBENT — never the checkpoint's own older snapshot."""
    model, tok = tiny_llama
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD)
    ckpt = tmp_path / "p" / "ckpt.pt"
    state = cont.load_state(ckpt)
    old_snapshot = [t.clone() for t in state["stopper"]["snapshot"]]
    fill = 0.321
    published = [torch.full_like(t, fill) for t in state["trainable"]]
    stopper = cont.incumbent_stopper(state, published_trainable=published,
                                     best_ce=0.0, best_epoch=2, since_best=0)
    assert stopper["epochs_seen"] == 1 and stopper["best_epoch"] == 2
    summ = cont.summarize(state, ckpt)
    cont.adopt(state, new_key=FP_NEW, dest_dir=tmp_path / "c",
               marker=cont.continuation_marker(summ, 6), source=ckpt,
               stopper=stopper)
    adopted = torch.load(tmp_path / "c" / "ckpt.pt", weights_only=False)
    assert adopted["continued_from"]["incumbent"] == {
        "source": "published_object", "best_epoch": 2, "best": 0.0,
        "since_best": 0}
    got, diag, _ = _fit(model, tok, epochs=6, ckdir=tmp_path / "c", key=FP_NEW,
                        patience=1)
    assert diag["resumed_from_step"] == 6
    assert diag["stopped_early"] is True and diag["epochs_run"] == 2
    assert diag["best_epoch"] == 2
    for t, old in zip(got.trainable, old_snapshot):
        assert torch.equal(t, torch.full_like(t, fill))
        assert not torch.equal(t, old)
    # shape mismatch refuses; a stopper-less checkpoint refuses
    with pytest.raises(cont.ContinuationError, match="count or shapes"):
        cont.incumbent_stopper(state, published_trainable=published[:-1],
                               best_ce=0.0, best_epoch=2, since_best=0)
    with pytest.raises(cont.ContinuationError, match="no stopper"):
        cont.incumbent_stopper({**state, "stopper": None},
                               published_trainable=published, best_ce=0.0,
                               best_epoch=2, since_best=0)


# --------------------------------------------------------------------------- #
# (c) refusals                                                                 #
# --------------------------------------------------------------------------- #


def test_load_state_refusals(tmp_path):
    with pytest.raises(cont.ContinuationError, match="does not exist"):
        cont.load_state(tmp_path / "nope.pt")
    p = tmp_path / "bad.pt"
    torch.save({"foo": 1}, p)
    with pytest.raises(cont.ContinuationError, match="lacks checkpoint field"):
        cont.load_state(p)
    st = _synthetic_state()
    st["trainer"] = "train_lockstep"
    torch.save(st, p)
    with pytest.raises(cont.ContinuationError, match="written by trainer"):
        cont.load_state(p)
    st = _synthetic_state()
    st["history"] = st["history"][:-1]
    torch.save(st, p)
    with pytest.raises(cont.ContinuationError, match="internally inconsistent"):
        cont.load_state(p)
    st = _synthetic_state()
    st["stopper"]["epochs_seen"] = 5
    torch.save(st, p)
    with pytest.raises(cont.ContinuationError, match="epochs_seen"):
        cont.load_state(p)
    st = _synthetic_state()
    st["heldout_curve"] = [1.0]
    torch.save(st, p)
    with pytest.raises(cont.ContinuationError, match="held-out curve"):
        cont.load_state(p)
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(cont.ContinuationError, match="unreadable"):
        cont.load_state(p)
    good = _synthetic_state()
    torch.save(good, p)
    s = cont.summarize(cont.load_state(p), p)
    assert s["checkpoint_key"] == FP_OLD and s["step"] == 6
    # a new marker carries the stopper rule; stopper_rule=None
    # derives the legacy marker an earlier publication was stamped with
    legacy = {"fit_fingerprint": FP_OLD, "step": 6, "epoch": 1,
              "batches_done": 2, "epochs_cap": 34}
    assert cont.continuation_marker(s, 34) == {
        **legacy, "stopper_rule": cont.STOPPER_RULE_REPLAY_EXEMPT}
    assert cont.continuation_marker(s, 34, stopper_rule=None) == legacy
    with pytest.raises(cont.ContinuationError, match=">= 1"):
        cont.continuation_marker(s, 0)
    with pytest.raises(cont.ContinuationError, match="did not change"):
        cont.adopt(good, new_key=FP_OLD, dest_dir=tmp_path / "d", marker={})


def test_verify_cell_binds_the_checkpoint_to_the_published_object(tmp_path):
    state = _synthetic_state()
    cells = tmp_path / "cells_all"
    # the real thing: object == the checkpoint's best snapshot, same best epoch
    d = _stage_cell(cells, state)
    ver = cont.verify_cell(d, state, expect_revision="r0")
    assert ver["problems"] == [], ver["problems"]
    assert ver["checks"]["object_matches_checkpoint_best_snapshot"] is True
    assert ver["checks"]["prior_fit_fingerprint"] == FP_OLD
    assert ver["checks"]["prior_epochs_run"] == 2
    assert ver["checks"]["object_sha256"] == _sha(d / "object.pt")
    # a recorded revision that differs refuses; a sidecar recording NO
    # revision is a stated absence, not a mismatch
    assert any("model_revision" in p for p in
               cont.verify_cell(d, state, expect_revision="r9")["problems"])
    shutil.rmtree(d)
    d = _stage_cell(cells, state, revision=None)
    assert cont.verify_cell(d, state, expect_revision="r9")["problems"] == []

    def problems(**kw):
        shutil.rmtree(cells, ignore_errors=True)
        dd = _stage_cell(cells, state, **kw)
        return cont.verify_cell(dd, state)["problems"]

    # a foreign fit's checkpoint (marker fingerprint != checkpoint key)
    assert any("not written by the fit" in p for p in problems(fp=FP_NEW))
    # a checkpoint past the published fit's last step postdates the object
    assert any("postdates" in p
               for p in problems(fit_record={"optimizer_steps_run": 5}))
    assert any("not inside" in p for p in problems(fit_record={"epochs_run": 1}))
    # the checkpoint's best epoch beyond the published one
    assert any("exceeds" in p for p in problems(fit_record={"best_epoch": 0}))
    # same best epoch, different tensors: not this object's checkpoint
    assert any("does not equal the published object" in p for p in problems(
        object_tensors=[torch.zeros_like(t) for t in state["trainable"]]))
    # the published best came AFTER the checkpoint: the
    # PUBLISHED object is the incumbent; since_best comes from the published
    # curve (2 epochs run, best at 2 -> 0 strikes after it)
    shutil.rmtree(cells)
    later = {"best_epoch": 2, "best_heldout_ce": 0.8,
             "heldout_curve": [1.5, 0.9, 0.8]}
    dd = _stage_cell(cells, state, fit_record=later,
                     object_tensors=[torch.zeros_like(t)
                                     for t in state["trainable"]])
    ver = cont.verify_cell(dd, state)
    assert ver["problems"] == [], ver["problems"]
    assert ver["checks"]["incumbent"] == "published_object"
    assert ver["checks"]["incumbent_best_epoch"] == 2
    assert ver["checks"]["incumbent_best_heldout_ce"] == 0.8
    assert ver["checks"]["incumbent_since_best"] == 0
    assert ver["checks"]["object_matches_checkpoint_best_snapshot"] is None
    assert "incumbent" in ver["checks"]["object_match_note"]
    # a best at epoch 2 of 3 run -> one strike already on the published record
    shutil.rmtree(cells)
    dd = _stage_cell(cells, state, fit_record={
        **later, "epochs_run": 3, "optimizer_steps_run": 12,
        "heldout_curve": [1.5, 0.9, 0.8, 0.85]})
    ver = cont.verify_cell(dd, state)
    assert ver["problems"] == [] and ver["checks"]["incumbent_since_best"] == 1
    # ... but never without the published curve, and never on a curve that
    # disagrees with its own best
    for bad in ({**later, "heldout_curve": None},
                {**later, "heldout_curve": [1.5, 0.9]},
                {**later, "heldout_curve": [1.5, 0.7, 0.8]},
                {**later, "heldout_curve": [1.5, 0.9, 0.81]},
                {**later, "best_heldout_ce": None}):
        shutil.rmtree(cells)
        dd = _stage_cell(cells, state, fit_record=bad)
        probs = cont.verify_cell(dd, state)["problems"]
        assert probs and all("refusing" in p for p in probs), (bad, probs)
    # the equal-best case keeps the checkpoint snapshot as incumbent
    shutil.rmtree(cells)
    dd = _stage_cell(cells, state)
    assert cont.verify_cell(dd, state)["checks"]["incumbent"] == \
        "checkpoint_snapshot"
    # a tampered object: marker + sidecar shas and the snapshot all disagree
    shutil.rmtree(cells)
    dd = _stage_cell(cells, state)
    with (dd / "object.pt").open("ab") as f:
        f.write(b"\0")
    probs = cont.verify_cell(dd, state)["problems"]
    assert any("published sha256" in p for p in probs)
    assert any("stale object/sidecar" in p for p in probs)
    # wrong cell: the dir is another cell's
    shutil.rmtree(cells)
    dd = _stage_cell(cells, state)
    other = cells / "constv2-mix-10k__lora_r4_ce"
    dd.rename(other)
    probs = cont.verify_cell(other, state)["problems"]
    assert any("COMPLETE marker names cell_id" in p for p in probs)
    assert any("sidecar records cell_id" in p for p in probs)
    # no marker at all
    shutil.rmtree(cells)
    dd = _stage_cell(cells, state)
    (dd / "COMPLETE").unlink()
    assert any("COMPLETE marker" in p and "missing" in p
               for p in cont.verify_cell(dd, state)["problems"])
    # the supersede record is read and judged against the CURRENT marker
    shutil.rmtree(cells)
    dd = _stage_cell(cells, state)
    (dd / "superseded_by.json").write_text(json.dumps(
        {"fit_fingerprint": FP_OLD, "snapshot": "/x/_r2cap24/cells_all/c"}))
    ver = cont.verify_cell(dd, state)
    assert ver["problems"] == []
    assert ver["checks"]["superseded_by"] == {
        "fit_fingerprint": FP_OLD, "snapshot": "/x/_r2cap24/cells_all/c",
        "licenses_current_publication": True}
    (dd / "superseded_by.json").write_text(json.dumps(
        {"fit_fingerprint": "c" * 64, "snapshot": "/older"}))
    assert cont.verify_cell(dd, state)["checks"]["superseded_by"][
        "licenses_current_publication"] is False


def test_driver_refusals_fire_before_any_training(tiny_llama, tmp_path):
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    cfg = _driver_cfg(tmp_path, epochs=2)
    out = tmp_path / "bank" / "cells_all"
    ck = tmp_path / "ckpt"
    pool_sha = t2._sha256_file(pool)
    base = dict(cfg=cfg, save_dir=out, pairs_path=str(pool),
                pool_sha256=pool_sha, config_sha="cfgsha",
                checkpoint_dir=ck, checkpoint_every=3)
    prior_fp = t2.cell_fit_fingerprint(cfg, SI, ARM, positions="all",
                                       pool_sha256=pool_sha,
                                       config_sha="cfgsha")
    cell = out / CELL
    (cell / "final_ckpt").mkdir(parents=True)
    ckpt = cell / "final_ckpt" / "ckpt.pt"
    torch.save(_synthetic_state(key=prior_fp), ckpt)

    def run(**over):
        kw = {**base, **over}
        return t2.train_si_objects(model, tok, SI, docs, kw.pop("arms", [ARM]),
                                   **kw)

    with pytest.raises(SystemExit, match="without --continue-from"):
        run(epochs_cap=4)
    with pytest.raises(SystemExit, match="without --epochs-cap"):
        run(continue_from=ckpt)
    with pytest.raises(SystemExit, match="must exceed the prior cap 2"):
        run(continue_from=ckpt, epochs_cap=2)
    with pytest.raises(SystemExit, match="below the prior cap 2"):
        run(continue_from=ckpt, epochs_cap=1)
    # the SAME cap with a DIFFERENT lr multiplier is a different fit and
    # passes the cap gate (it then fails the publication check here, since
    # this cell dir holds no publication) — the re-issue of a continuation
    # that early-stopped at ~peak lr, at the same cap with a gentler peak
    # (at the SAME cap this checkpoint sits inside the cap's final
    # epoch — under replay_exempt the only epoch that would run is the
    # replayed, unjudged one, so the cap gate refuses before verify_cell)
    with pytest.raises(SystemExit, match="no epoch would be judged"):
        run(continue_from=ckpt, epochs_cap=2, continue_lr_mult=0.25)
    # the lr multiplier belongs to a continuation only
    with pytest.raises(SystemExit, match="without --continue-from"):
        run(continue_lr_mult=0.5)
    with pytest.raises(SystemExit, match="finite positive multiplier"):
        run(continue_from=ckpt, epochs_cap=4, continue_lr_mult=0.0)
    # the CLI keeps an explicit 0 so that refusal fires (a truthiness
    # default turned `--continue-lr-mult 0` into a full-rate restart
    # under the ordinary fingerprint)
    assert t2._continue_lr_mult_arg(SimpleNamespace(continue_lr_mult=0.0)) == 0.0
    assert t2._continue_lr_mult_arg(SimpleNamespace(continue_lr_mult=None)) == 1.0
    assert t2._continue_lr_mult_arg(SimpleNamespace()) == 1.0
    assert t2._continue_lr_mult_arg(SimpleNamespace(continue_lr_mult=0.25)) == 0.25
    with pytest.raises(SystemExit, match="finite positive multiplier"):
        run(continue_from=ckpt, epochs_cap=4,
            continue_lr_mult=t2._continue_lr_mult_arg(
                SimpleNamespace(continue_lr_mult=0)))
    with pytest.raises(SystemExit, match="finite positive multiplier"):
        run(continue_from=ckpt, epochs_cap=4, continue_lr_mult=float("nan"))
    with pytest.raises(SystemExit, match="exactly ONE document-CE arm"):
        run(continue_from=ckpt, epochs_cap=4, arms=[ARM, "psr_ce_all"])
    with pytest.raises(SystemExit, match="needs --checkpoint-dir"):
        run(continue_from=ckpt, epochs_cap=4, checkpoint_dir=None)
    # the right key, but the cell dir holds no publication to verify against
    with pytest.raises(SystemExit, match="does not belong to the published"):
        run(continue_from=ckpt, epochs_cap=4)
    # a checkpoint outside <cell>/final_ckpt/
    stray = tmp_path / "stray.pt"
    torch.save(_synthetic_state(key=prior_fp), stray)
    with pytest.raises(SystemExit, match="not <cell>/final_ckpt"):
        run(continue_from=stray, epochs_cap=4)
    # a checkpoint whose identity is not THIS config's (anything but the cap
    # differs — here the checkpoint's key is another fit's)
    torch.save(_synthetic_state(key="z" * 64), ckpt)
    with pytest.raises(SystemExit, match="EXACTLY the prior fit's identity"):
        run(continue_from=ckpt, epochs_cap=4)
    # the same refusal when the CONFIG moved (a different lr): the prior
    # fingerprint no longer resolves to the checkpoint's key
    torch.save(_synthetic_state(key=prior_fp), ckpt)
    with pytest.raises(SystemExit, match="EXACTLY the prior fit's identity"):
        run(continue_from=ckpt, epochs_cap=4,
            cfg=_driver_cfg(tmp_path, epochs=2, lr=2e-3))
    # nothing was published or trained by any of the above
    assert not (cell / "COMPLETE").exists()
    assert not (ck / CELL).exists()


# --------------------------------------------------------------------------- #
# (d) snapshot + supersede + the in-container skip                             #
# --------------------------------------------------------------------------- #


def test_snapshot_r2cap24_carries_final_ckpt_and_reports_and_licenses(sf,
                                                                      tmp_path):
    state = _synthetic_state()
    bank = tmp_path / "out" / "key"
    cells = bank / "cells_all"
    cells.mkdir(parents=True)
    (cells / sc.STORE_IDENT).write_text(json.dumps(
        {"model": "fake/base", "model_revision": "r0", "positions": "all"}))
    cell = _stage_cell(cells, state)
    (cell / "final_ckpt").mkdir()
    torch.save(state, cell / "final_ckpt" / "ckpt.pt")
    (cell / "final_ckpt" / "ckpt.prev.pt").write_bytes(b"prev")
    (cell / "object.pt.provenance.json").write_text('{"kind": "synthetic"}')
    (bank / "reports").mkdir()
    (bank / "reports" / f"report_{CELL}_all.json").write_text('{"arms": {}}')
    (bank / "reports" / f"metrics_{CELL}_all.jsonl").write_text('{"step": 1}\n')
    # the wave-A supersede record of an OLDER publication sits there too
    (cell / "superseded_by.json").write_text(json.dumps(
        {"cell_id": CELL, "fit_fingerprint": "c" * 64, "snapshot": "/old_r1"}))

    # BEFORE the snapshot: the continuation is refused as an unsnapshotted
    # supersede (the record names an older publication, not this one)
    assert "never snapshotted" in sf.supersede_gate(cell, FP_NEW)

    assert sf.snapshot_bank_root("out/key", "_r2cap24") == "out/key_r2cap24"
    assert sf.snapshot_bank_root("out/key") == "out/key_r1"
    with pytest.raises(ValueError, match="suffix"):
        sf.snapshot_bank_root("out/key", "r2cap24")
    dst = tmp_path / "out" / sf.snapshot_bank_root("key", "_r2cap24")
    rec = sc.snapshot_cell(bank, dst, CELL, superseded_by={
        "reason": "R2-D20 cap 24 -> 34 continuation",
        "expected_refit_fingerprint": FP_NEW, "continue_cap": 34})
    snap = dst / "cells_all" / CELL
    # the whole publication travelled: object, sidecars, metrics, the
    # retained checkpoint pair, the PSR-style provenance sidecar
    for name in ("object.pt", "train_budget_meta.json", "metrics.jsonl",
                 "COMPLETE", "object.pt.provenance.json",
                 "final_ckpt/ckpt.pt", "final_ckpt/ckpt.prev.pt"):
        assert (snap / name).read_bytes() == (cell / name).read_bytes(), name
    assert not (snap / "superseded_by.json").exists()  # never copied
    assert rec["final_ckpt_retained"] is True
    # ... and the bank-level report/metrics tees
    assert set(rec["reports"]) == {f"reports/report_{CELL}_all.json",
                                   f"reports/metrics_{CELL}_all.jsonl"}
    assert (dst / "reports" / f"report_{CELL}_all.json").read_text() == \
        '{"arms": {}}'
    on_disk = json.loads((snap / sc.SNAPSHOT_RECORD).read_text())
    assert on_disk["reports"] == rec["reports"]
    assert on_disk["final_ckpt_retained"] is True
    # the source-side record names the cap-M publication AND the
    # continuation as its successor — what the analysis' refit detector and
    # the supersede gate read
    sup = json.loads((cell / "superseded_by.json").read_text())
    assert sup["fit_fingerprint"] == FP_OLD
    assert sup["snapshot"] == str(snap)
    assert sup["superseded_by"] == {
        "reason": "R2-D20 cap 24 -> 34 continuation",
        "expected_refit_fingerprint": FP_NEW, "continue_cap": 34}
    # AFTER: exactly the continuation is licensed; the snapshot verifies
    assert sf.supersede_gate(cell, FP_NEW) is None
    sc.verify_snapshot(dst, CELL)
    # idempotent re-snapshot (identical) is a no-op, reports included
    rec2 = sc.snapshot_cell(bank, dst, CELL, superseded_by={"reason": "again"})
    assert rec2["already_present"] is True and rec2["reports"] == rec["reports"]
    # a differing report in the snapshot bank refuses (never overwritten)
    (dst / "reports" / f"report_{CELL}_all.json").write_text('{"arms": {"x": 1}}')
    with pytest.raises(sc.SnapshotError, match="never overwritten"):
        sc.snapshot_cell(bank, dst, CELL, superseded_by={"reason": "again"})


def test_snapshot_mode_and_impl_thread_suffix_and_continuation(sf):
    entry = SF_SRC.split("def main(", 1)[1]
    snap = entry.split('if mode == "snapshot"', 1)[1].split('if mode == "', 1)[0]
    assert "bool(verify_only), reason" in snap
    assert 'bank_suffix or "_r1"' in snap and "int(continue_cap or 0)" in snap
    impl = SF_SRC.split("def _snapshot_impl", 1)[1].split("\ndef ", 1)[0]
    assert "snapshot_bank_root(out_root, suffix)" in impl
    assert "continue_cap=cont_cap or None" in impl
    assert "retained no" in impl and "SnapshotError(" in impl  # refuses w/o ckpt
    # the volume is reloaded BEFORE either branch reads it (the cont2
    # --verify-only stale-view wart), and a continuation identity the
    # container would refuse refuses the snapshot
    assert "vol.reload()" in impl
    assert impl.index("vol.reload()") < impl.index("if verify_only:")
    assert "continuation identity refused" in impl
    assert "item[7] is not None" in impl
    assert "int(continue_cap or 0), lr_mult)" in snap
    sig = SF_SRC.split("def main(", 1)[1].split("):", 1)[0]
    assert "continue_cap: int = 0" in sig and 'bank_suffix: str = ""' in sig
    # the docstring documents the two operator steps
    head = SF_SRC.split("\nimport json", 1)[0]
    assert "--bank-suffix _r2cap24 --continue-cap 34" in head
    assert "--continue-cap 34 --manifest" in head


# --------------------------------------------------------------------------- #
# (e) watcher + fits-spawn + the app's fingerprint derivation                  #
# --------------------------------------------------------------------------- #


def _row(**over):
    c = {"model": "gemma4e4b_base", "si": "constv2-mix-1m", "arm": "psr_ce_all",
         "positions": "all", "leg": "all",
         "cell_id": "constv2-mix-1m__psr_ce_all",
         "out_root": "out/gemma4e4b_base", "expect_fingerprint": FP_NEW,
         "app": "sdfbase-fits", "function": "fit_h200_std",
         "function_id": "fu-deployed-1", "config_sha": "abcdef0123456789",
         "overwrite": False, "checkpoint_every": 100, "continue_cap": 34,
         "continue_lr_mult": 1.0, "app_id": "ap-1", "app_version": 11,
         "deploy_commit": "ea7a058ac6e0",
         "continued_from": {"prior_fingerprint": FP_OLD, "resume_step": 1200,
                            "resume_epoch": 23, "resume_batches_done": 27,
                            "n_batches": 51},
         "call_id": "fc-1", "spawn_time": "2026-01-01T00:00:00+00:00",
         "status": "spawned (in-container skip expected)"}
    c.update(over)
    return c


def _marker(fp):
    return json.dumps({"cell_id": "constv2-mix-1m__psr_ce_all",
                       "fit_fingerprint": fp,
                       "files": {"object.pt": "0" * 64}}).encode()


def test_watcher_validates_the_continued_row_and_rebuilds_its_item(fw):
    row = _row()

    def assess(marker, call):
        return fw.assess([row], strict=True, read_marker=lambda c: marker,
                         call_state=lambda c: call)

    # the cap-24 marker under the running continuation: in flight, not done
    res = assess(_marker(FP_OLD), "running")
    assert res["n_complete"] == 0 and res["all_calls_terminal"] is False
    assert res["needs_attention"] == []
    assert res["lines"]["gemma4e4b_base/constv2-mix-1m__psr_ce_all/all"] \
        .endswith("running (marker stale)")
    # the cap-24 marker under an ENDED call: triage, never complete
    res = assess(_marker(FP_OLD), "done")
    assert res["n_complete"] == 0 and len(res["needs_attention"]) == 1
    assert "!= expected" in res["needs_attention"][0]
    # only the CONTINUED publication counts
    res = assess(_marker(FP_NEW), "done")
    assert res["n_complete"] == 1 and res["needs_attention"] == []
    # the successor of a continuation is the same continuation (8-tuple)
    assert fw.successor_item(row) == (
        "gemma4e4b_base", "constv2-mix-1m", "psr_ce_all", "abcdef0123456789",
        False, "out/gemma4e4b_base", 100, 34)
    # plain rows keep the 7-tuple byte-for-byte
    plain = _row(continue_cap=0)
    plain.pop("continued_from")
    assert fw.successor_item(plain) == (
        "gemma4e4b_base", "constv2-mix-1m", "psr_ce_all", "abcdef0123456789",
        False, "out/gemma4e4b_base", 100)
    legacy = {k: v for k, v in _row().items() if k != "continue_cap"}
    assert fw.successor_item(legacy) == fw.successor_item(plain)
    assert "continue_cap" in fw.ROW_PIN_FIELDS
    assert "continue_lr_mult" in fw.ROW_PIN_FIELDS
    # a non-1.0 lr multiplier is a 9th element; 1.0 / absent keeps the
    # 8-tuple byte for byte (the cont2 rows)
    assert fw.successor_item(_row(continue_lr_mult=0.25)) == (
        "gemma4e4b_base", "constv2-mix-1m", "psr_ce_all", "abcdef0123456789",
        False, "out/gemma4e4b_base", 100, 34, 0.25)
    no_mult = {k: v for k, v in _row().items() if k != "continue_lr_mult"}
    assert fw.successor_item(no_mult) == fw.successor_item(row)
    ok, why = fw.can_respawn(row, "timeout", max_successors=3, marker="stale")
    assert ok, why


def test_continuation_cells_selector(sf):
    assert sf.continuation_cells(
        "constv2-mix-1m__psr_ce_all,constv2-mix-1m__lora_r4_ce", cap=34) == [
        ("constv2-mix-1m", "psr_ce_all"), ("constv2-mix-1m", "lora_r4_ce")]
    with pytest.raises(SystemExit, match="needs --cells"):
        sf.continuation_cells("", cap=34)
    with pytest.raises(SystemExit, match="--cells only"):
        sf.continuation_cells("constv2-mix-1m__psr_ce_all", sis="constv2-mix-1m",
                              cap=34)
    with pytest.raises(SystemExit, match="--cells only"):
        sf.continuation_cells("constv2-mix-1m__psr_ce_all", retrain_all=True,
                              cap=34)
    with pytest.raises(SystemExit, match="overwrite"):
        sf.continuation_cells("constv2-mix-1m__psr_ce_all", overwrite=True,
                              cap=34)
    with pytest.raises(SystemExit, match=">= 1"):
        sf.continuation_cells("constv2-mix-1m__psr_ce_all", cap=0)
    with pytest.raises(SystemExit, match="not a <si>__<arm>"):
        sf.continuation_cells("constv2-mix-1m", cap=34)
    with pytest.raises(SystemExit, match="unknown cell"):
        sf.continuation_cells("constv2-mix-1b__psr_ce_all", cap=34)
    with pytest.raises(SystemExit, match="named twice"):
        sf.continuation_cells("constv2-mix-1m__psr_ce_all,"
                              "constv2-mix-1m__psr_ce_all", cap=34)


def test_fits_spawn_continue_cap_writes_continuation_rows(sf, fake, tmp_path):
    """The operator's fits-spawn --continue-cap, end to end under the modal
    stub: fingerprint_cpu is asked for CONTINUED fingerprints (item carries
    the cap + out_root), every row records the continuation pin + the
    continued_from block + the continued fit's expect_fingerprint, every
    spawned item is the 8-tuple the watcher's successor_item rebuilds, and
    the refusals hold."""
    cfg_local = sf.WORKTREE / Path(
        sf.MODELS["gemma4e4b_base"]["cfg"]).relative_to("/repo")
    sha = sf._config_sha16(cfg_local)
    seen: dict = {}

    def remote(item):
        seen["item"] = item
        key, pairs, cap, root = item[:4]
        cells = [f"{s}__{a}" for s, a in pairs]
        return {"model": key, "config_sha": sha, "missing_pools": [],
                "continue_cap": cap,
                "fingerprints": {c: FP_NEW for c in cells},
                "continuations": {c: {"prior_fingerprint": FP_OLD,
                                      "resume_step": 1200, "resume_epoch": 23,
                                      "resume_batches_done": 27,
                                      "n_batches": 51} for c in cells},
                "missing_checkpoints": []}

    fake.remote = remote
    # the deployment pin as fits-spawn reads it live
    sf._deployment_pin = lambda app: {"app_id": "ap-1", "app_version": 3,
                                      "deploy_commit": "abc123def456",
                                      "app_name_seen": app}
    mpath = tmp_path / "fit_calls_cont.json"
    cells = "constv2-mix-1m__psr_ce_all,constv2-mix-1m__lora_r4_ce"
    sf.main(mode="fits-spawn", model="gemma4e4b_base", cells=cells,
            continue_cap=34, manifest=str(mpath), checkpoint_every=100)
    assert seen["item"] == ("gemma4e4b_base",
                            [["constv2-mix-1m", "psr_ce_all"],
                             ["constv2-mix-1m", "lora_r4_ce"]],
                            34, "out/gemma4e4b_base")   # 4-tuple at m=1.0
    doc = json.loads(mpath.read_text())
    assert doc["continue_cap"] == 34 and doc["function"] == "fit_h200_std"
    assert doc["deployment"]["app_version"] == 3
    assert doc["deployment"]["app_name_seen"] == "sdfbase-fits"
    rows = doc["cells"]
    assert [r["cell_id"] for r in rows] == cells.split(",")
    for r in rows:
        assert r["continue_cap"] == 34 and r["continue_lr_mult"] == 1.0
        assert r["expect_fingerprint"] == FP_NEW
        assert r["continued_from"]["prior_fingerprint"] == FP_OLD
        assert r["continued_from"]["resume_step"] == 1200
        assert r["function_id"] == "fu-deployed-1"
        assert (r["app_id"], r["app_version"], r["deploy_commit"]) == (
            "ap-1", 3, "abc123def456")
        assert r["config_sha"] == sha and r["overwrite"] is False
        assert r["call_id"] and r["status"].startswith("spawned")
    assert fake.spawned == [
        ("gemma4e4b_base", "constv2-mix-1m", "psr_ce_all", sha, False,
         "out/gemma4e4b_base", 100, 34),
        ("gemma4e4b_base", "constv2-mix-1m", "lora_r4_ce", sha, False,
         "out/gemma4e4b_base", 100, 34)]
    fw = _load_under_fake(FW_PATH, "_fw_for_spawn_rows", fake)
    for r, item in zip(rows, fake.spawned):
        assert fw.successor_item(r) == item
    # --continue-lr-mult: the remote item and the spawned item carry it,
    # the row pins it, successor_item rebuilds it
    fake.spawned.clear()
    sf.main(mode="fits-spawn", model="gemma4e4b_base",
            cells="constv2-mix-1m__lora_r4_ce", continue_cap=60,
            continue_lr_mult=0.25, manifest=str(mpath), checkpoint_every=100)
    assert seen["item"][2:] == (60, "out/gemma4e4b_base", 0.25)
    assert fake.spawned == [
        ("gemma4e4b_base", "constv2-mix-1m", "lora_r4_ce", sha, False,
         "out/gemma4e4b_base", 100, 60, 0.25)]
    doc = json.loads(mpath.read_text())
    r60 = [r for r in doc["cells"] if r["continue_cap"] == 60]
    assert len(r60) == 1 and r60[0]["continue_lr_mult"] == 0.25
    assert fw.successor_item(r60[0]) == fake.spawned[0]
    # the psr row of the first spawn survives the merge
    assert [r["continue_cap"] for r in doc["cells"]] == [34, 60]
    with pytest.raises(SystemExit, match="without --continue-cap"):
        sf.main(mode="fits-spawn", model="gemma4e4b_base",
                sis="constv2-mix-1m", continue_lr_mult=0.5,
                manifest=str(mpath))
    with pytest.raises(SystemExit, match="finite positive"):
        sf.main(mode="fits-spawn", model="gemma4e4b_base", cells=cells,
                continue_cap=60, continue_lr_mult=-1.0, manifest=str(mpath))
    # when the deployment cannot be read the row is pin-less and the
    # watcher will refuse its successors (a warning, not a silent pass)
    def _boom(app):
        raise RuntimeError("no deployment history")
    sf._deployment_pin = _boom
    fake.spawned.clear()
    sf.main(mode="fits-spawn", model="gemma4e4b_base",
            cells="constv2-mix-1m__psr_ce_all", continue_cap=34,
            manifest=str(mpath), checkpoint_every=100)
    doc = json.loads(mpath.read_text())
    unpinned = [r for r in doc["cells"] if r["cell_id"].endswith("psr_ce_all")]
    assert unpinned[0]["app_version"] is None and doc["deployment"] is None
    # refusals — nothing spawns
    n = len(fake.spawned)
    with pytest.raises(SystemExit, match="--cells only"):
        sf.main(mode="fits-spawn", model="gemma4e4b_base", sis="constv2-mix-1m",
                continue_cap=34, manifest=str(mpath))
    with pytest.raises(SystemExit, match="overwrite"):
        sf.main(mode="fits-spawn", model="gemma4e4b_base", cells=cells,
                continue_cap=34, overwrite=True, manifest=str(mpath))
    with pytest.raises(SystemExit, match="needs --cells"):
        sf.main(mode="fits-spawn", model="gemma4e4b_base", continue_cap=34,
                manifest=str(mpath))
    fake.remote = lambda item: {
        "model": item[0], "config_sha": sha, "missing_pools": [],
        "fingerprints": {}, "continuations": {},
        "missing_checkpoints": ["constv2-mix-1m__psr_ce_all: absent"]}
    with pytest.raises(SystemExit, match="no continuable retained checkpoint"):
        sf.main(mode="fits-spawn", model="gemma4e4b_base",
                cells="constv2-mix-1m__psr_ce_all", continue_cap=34,
                manifest=str(mpath))
    assert len(fake.spawned) == n


def test_fit_impl_and_fingerprint_impl_thread_the_continuation():
    impl = SF_SRC.split("def _fit_impl", 1)[1].split("\ndef ", 1)[0]
    assert "cont_cap = int(item[7]) if len(item) > 7 and item[7] else 0" in impl
    assert '"status": "refused-no-final-checkpoint"' in impl
    assert '"status": "refused-continue-with-overwrite"' in impl
    assert '"--continue-from", str(final_ckpt)' in impl
    assert '"--epochs-cap", str(cont_cap)' in impl
    # the refusals precede the derivation, which precedes the gate + seed
    i_ref = impl.index("refused-no-final-checkpoint")
    i_fp = impl.index("expected_fit_fingerprint(")
    i_gate = impl.index("supersede_gate(vol_out / cell_id, expect_fp)")
    i_seed = impl.index("shutil.copytree(vol_out / cell_id")
    assert i_ref < i_fp < i_gate < i_seed
    assert "continue_cap=cont_cap or None" in impl
    fp_impl = SF_SRC.split("def _fingerprint_impl", 1)[1].split("\ndef ", 1)[0]
    assert "cont_cap = int(item[2])" in fp_impl
    assert 'final_ckpt/ckpt.pt' in fp_impl and '"missing_checkpoints"' in fp_impl
    assert "continuation_summary(ckpt)" in fp_impl
    entry = SF_SRC.split('if mode == "fits-spawn"', 1)[1].split(
        'if mode == "watch"', 1)[0]
    assert "fp_req = (key, [[s, a] for s, a in fp_pairs], cont_cap, root)" \
        in entry                                  # the remote item carries both
    assert "fp_req += (lr_mult,)" in entry         # …and a non-1.0 multiplier
    assert "item += (cont_cap,)" in entry
    assert "item += (lr_mult,)" in entry
    # the container's derivation PROVES the chain and a refusal is a
    # deterministic `done` (never a respawn cause); the lr multiplier is
    # the 9th item element and reaches the driver
    assert ("cont_lr_mult = (float(item[8]) if len(item) > 8 and item[8] is "
            "not None") in impl                    # an explicit 0 is refused
    assert '"status": "refused-continuation-identity"' in impl
    # the fit container reloads the volume BEFORE its first /vol read (a
    # warm container's stale snapshot could re-adopt final_ckpt over a
    # successor's newer transient checkpoint)
    assert "vol.reload()" in impl
    assert impl.index("vol.reload()") < impl.index("_ensure_model(key)")
    assert impl.index("vol.reload()") < impl.index("os.path.exists(pairs)")
    assert impl.index("vol.reload()") < impl.index("final_ckpt.is_file()")
    assert '"--continue-lr-mult", repr(float(cont_lr_mult))' in impl
    assert "item[4] is not None" in fp_impl
    assert "vol.reload()" in fp_impl
    assert "expected_continuation(" in fp_impl
    assert fp_impl.index("vol.reload()") < fp_impl.index("final_ckpt/ckpt.pt")


def test_expected_fit_fingerprint_with_continuation_is_the_drivers(sf, tmp_path):
    """The app's in-container derivation == the driver's cell_fit_fingerprint
    with the SAME continuation marker; a different cap or resume position is
    a different identity; the plain fingerprint differs from the continued
    one (the cap-M COMPLETE can never skip the continuation)."""
    cfg = _driver_cfg(tmp_path, epochs=2)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg))       # JSON is valid YAML
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(_docs(), pool)
    ck = tmp_path / CELL / "final_ckpt"
    ck.mkdir(parents=True)
    fp_plain = sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM)
    # a first-generation checkpoint carries the PLAIN identity as its key
    torch.save(_synthetic_state(key=fp_plain), ck / "ckpt.pt")
    fp_cont = sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM,
                                          continue_cap=4,
                                          checkpoint=ck / "ckpt.pt")
    assert fp_plain != fp_cont
    loaded = t2._cfg_with_positions(t2._load_cfg(str(cfg_path)), "all")
    pool_sha, config_sha = t2._cli_fit_identity(
        SimpleNamespace(pairs=str(pool), config=str(cfg_path)), loaded)
    marker = cont.continuation_marker(
        cont.summarize(cont.load_state(ck / "ckpt.pt")), 4)
    assert fp_cont == t2.cell_fit_fingerprint(
        loaded, SI, ARM, positions="all", pool_sha256=pool_sha,
        config_sha=config_sha, teacher="online", continued_from=marker)
    assert fp_plain == t2.cell_fit_fingerprint(
        loaded, SI, ARM, positions="all", pool_sha256=pool_sha,
        config_sha=config_sha, teacher="online")
    exp = sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=4,
                                   checkpoint=ck / "ckpt.pt")
    assert exp["fingerprint"] == fp_cont and exp["marker"] == marker
    assert exp["chain"] == [marker] and exp["chain_depth"] == 1
    assert exp["prior_cap"] is None and exp["root_fit_fingerprint"] == fp_plain
    assert exp["lr_mult"] == 1.0 and exp["prior_lr_mult"] == 1.0
    assert sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=5,
                                       checkpoint=ck / "ckpt.pt") != fp_cont
    # the lr multiplier is identity; 1.0 leaves it alone
    fp_m = sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=4,
                                       checkpoint=ck / "ckpt.pt",
                                       continue_lr_mult=0.25)
    assert fp_m != fp_cont
    assert sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=4,
                                       checkpoint=ck / "ckpt.pt",
                                       continue_lr_mult=1.0) == fp_cont
    torch.save(_synthetic_state(key=fp_plain, step=7, bd=3), ck / "ckpt.pt")
    assert sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=4,
                                       checkpoint=ck / "ckpt.pt") != fp_cont
    summ = sf.continuation_summary(ck / "ckpt.pt")
    assert summ["prior_fingerprint"] == fp_plain and summ["resume_step"] == 7
    with pytest.raises(cont.ContinuationError, match="does not exist"):
        sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=4,
                                    checkpoint=tmp_path / "missing.pt")
    # a key that is NOT this identity and has no publication chain behind
    # it is REFUSED here — the container's driver would refuse it after
    # model load, so the pre-spawn derivation must
    torch.save(_synthetic_state(key=FP_OLD), ck / "ckpt.pt")
    with pytest.raises(cont.ContinuationError,
                       match="not the plain identity"):
        sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=4,
                                    checkpoint=ck / "ckpt.pt")


# --------------------------------------------------------------------------- #
# (f) end to end through the driver                                            #
# --------------------------------------------------------------------------- #


def _leaves(path: Path) -> list:
    out: list = []
    cont._leaves(torch.load(path, map_location="cpu", weights_only=True), out)
    return out


def test_driver_end_to_end_cap_raise_continuation(tiny_llama, tmp_path):
    """cap 2 publish -> retain final_ckpt/ (what _fit_impl does) -> snapshot
    into an _r2cap24 bank -> continue to cap 4 through train_si_objects:
    the cap-2 COMPLETE never skips it, the continued cell publishes under
    the continuation fingerprint with continued_from in sidecar + report +
    marker, the trainer resumed from the retained step, a respawn WOULD skip
    it now, and under constant lr the object equals the straight cap-4 fit's
    bitwise."""
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    pool_sha = t2._sha256_file(pool)
    cfg = _driver_cfg(tmp_path, epochs=2)
    bank = tmp_path / "bank"
    out = bank / "cells_all"
    ck = tmp_path / "ckpt"
    kw = dict(cfg=cfg, save_dir=out, pairs_path=str(pool), pool_sha256=pool_sha,
              config_sha="cfgsha", checkpoint_dir=ck, checkpoint_every=3)
    o1 = t2.train_si_objects(model, tok, SI, docs, [ARM], **kw)
    assert o1.arms[ARM].status == "trained", o1.arms[ARM]
    cell = out / CELL
    old_marker = json.loads((cell / "COMPLETE").read_text())
    fp_old = old_marker["fit_fingerprint"]
    assert "continued_from" not in old_marker
    # retention, as _fit_impl does it at publish: the transient checkpoint
    # pair moves into <cell>/final_ckpt/
    (cell / "final_ckpt").mkdir()
    for f in sorted((ck / CELL).iterdir()):
        shutil.move(str(f), str(cell / "final_ckpt" / f.name))
    shutil.rmtree(ck / CELL)
    ckpt = cell / "final_ckpt" / "ckpt.pt"
    state = cont.load_state(ckpt)
    assert state["checkpoint_key"] == fp_old
    summ = cont.summarize(state, ckpt)
    assert (summ["step"], summ["epoch"], summ["batches_done"]) == (6, 1, 2)
    # the REAL publication verifies against its own retained checkpoint
    ver = cont.verify_cell(cell, state)
    assert ver["problems"] == [], ver["problems"]
    assert ver["checks"]["prior_epochs_run"] == 2
    # the continuation identity differs; the cap-2 COMPLETE cannot skip it
    marker = cont.continuation_marker(summ, 4)
    fp_new = t2.cell_fit_fingerprint(cfg, SI, ARM, positions="all",
                                     pool_sha256=pool_sha, config_sha="cfgsha",
                                     continued_from=marker)
    assert fp_new != fp_old
    assert t2.validated_complete(out, CELL, fp_old) == cell / "object.pt"
    assert t2.validated_complete(out, CELL, fp_new) is None
    # snapshot BEFORE the overwrite (the operator's --mode snapshot
    # --bank-suffix _r2cap24 --continue-cap), naming the continuation
    snap_bank = tmp_path / "bank_r2cap24"
    rec = sc.snapshot_cell(bank, snap_bank, CELL, superseded_by={
        "reason": "R2-D20 test", "expected_refit_fingerprint": fp_new,
        "continue_cap": 4})
    snap = snap_bank / "cells_all" / CELL
    assert rec["final_ckpt_retained"] is True
    assert (snap / "final_ckpt" / "ckpt.pt").read_bytes() == ckpt.read_bytes()
    assert (snap / "object.pt").read_bytes() == (cell / "object.pt").read_bytes()
    sup = json.loads((cell / "superseded_by.json").read_text())
    assert sup["fit_fingerprint"] == fp_old
    assert sup["superseded_by"]["expected_refit_fingerprint"] == fp_new

    # CONTINUE: cap 2 -> 4 (the driver flags, as the container passes them)
    o2 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=4,
                             continue_from=ckpt, **kw)
    assert o2.arms[ARM].status == "trained", o2.arms[ARM]
    new_marker = json.loads((cell / "COMPLETE").read_text())
    assert new_marker["fit_fingerprint"] == fp_new
    cf = new_marker["continued_from"]
    assert cf["prior_fit_fingerprint"] == fp_old
    assert cf["prior_object_sha256"] == old_marker["object_sha256"]
    assert (cf["prior_cap"], cf["new_cap"]) == (2, 4)
    assert cf["prior_epochs_run"] == 2
    assert (cf["resume_step"], cf["resume_epoch"], cf["resume_batches_done"]) \
        == (6, 1, 2)
    assert cf["epochs_run_total"] == 4
    assert cf["epochs_from_resume_epoch_inclusive"] == 3
    assert cf["stop_reason"] == "cap"
    assert cf["resumed_from_step"] == 6 and cf["optimizer_steps_run"] == 16
    assert cf["adoption"]["adopted"] is True
    assert cf["lr_schedule"]["lr_schedule"] == "constant"
    assert cf["lr_schedule"]["lr_jump_fraction_of_peak"] == 0
    assert cf["snapshot"] == str(snap)
    assert cf["identity_checks"]["superseded_by"][
        "licenses_current_publication"] is True
    assert cf["checkpoint"] == str(ckpt)
    # which incumbent was seated follows from whether the published best
    # postdates the checkpoint (epoch index 1 -> checkpoint best epoch 1)
    prior_best = json.loads((snap / "train_budget_meta.json").read_text())[
        "fit_record"]["best_epoch"]
    assert cf["incumbent"]["source"] == (
        "published_object" if prior_best > 1 else "checkpoint_snapshot")
    assert cf["incumbent"]["best_epoch"] == prior_best
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    assert meta["continued_from"] == cf
    assert meta["fit_record"]["continued_from"] == cf
    assert meta["fit_record"]["epochs_run"] == 4
    report = json.loads(t2.report_path_for(out, SI).read_text())
    assert report["arms"][ARM]["detail"]["continued_from"] == cf
    # the marker's manifest still validates and a respawn WOULD now skip
    assert t2.validated_complete(out, CELL, fp_new) == cell / "object.pt"
    assert t2.validated_complete(out, CELL, fp_old) is None
    # the continued fit's own last checkpoint is what retention would keep
    assert (ck / CELL / "ckpt.pt").is_file()
    assert torch.load(ck / CELL / "ckpt.pt", weights_only=False)["step"] == 15

    # the straight cap-4 fit publishes the SAME object (constant lr)
    cfg4 = _driver_cfg(tmp_path, epochs=4)
    out4 = tmp_path / "bank4" / "cells_all"
    o3 = t2.train_si_objects(model, tok, SI, docs, [ARM], cfg=cfg4,
                             save_dir=out4, pairs_path=str(pool),
                             pool_sha256=pool_sha, config_sha="cfgsha4",
                             checkpoint_dir=tmp_path / "ckpt4",
                             checkpoint_every=3)
    assert o3.arms[ARM].status == "trained", o3.arms[ARM]
    a = cont._tensor_multiset(_leaves(cell / "object.pt"))
    b = cont._tensor_multiset(_leaves(out4 / CELL / "object.pt"))
    assert a == b, "continued object != straight cap-4 object"
    meta4 = json.loads((out4 / CELL / "train_budget_meta.json").read_text())
    assert meta4["fit_record"]["heldout_curve"] == \
        meta["fit_record"]["heldout_curve"]
    assert meta4["fit_record"]["best_epoch"] == meta["fit_record"]["best_epoch"]
    assert "continued_from" not in meta4


def _seat_unbeatable_published_best(cell: Path) -> None:
    """Test-only: make the PUBLISHED fit's best held-out CE 0.0 (unbeatable)
    in the sidecar's fit_record — curve point + best value, so verify_cell's
    self-consistency check still passes — and re-stamp the sidecar's sha in
    the COMPLETE manifest so the publication still verifies. A stand-in for
    an incumbent no continued epoch can beat (the cont2/cont3 cells)."""
    meta_p = cell / "train_budget_meta.json"
    meta = json.loads(meta_p.read_text())
    fr = meta["fit_record"]
    be = int(fr["best_epoch"])
    fr["best_heldout_ce"] = 0.0
    fr["best_heldout_loss"] = 0.0
    fr["heldout_curve"][be] = 0.0          # index 0 is the pre-training point
    meta_p.write_text(json.dumps(meta))
    m_p = cell / "COMPLETE"
    m = json.loads(m_p.read_text())
    m["files"]["train_budget_meta.json"] = _sha(meta_p)
    m_p.write_text(json.dumps(m))


def test_driver_ships_published_best_when_no_full_epoch_beats_it(tiny_llama,
                                                                tmp_path):
    """End to end: the cap-2 fit's best is its FINAL
    epoch (after the retained mid-epoch checkpoint) and is seated unbeatable
    (sidecar edit, sha-consistent); patience is 1. The continuation's REPLAY
    of epoch 2 is exempt — logged, kept in the curve, flagged, NOT a strike
    — so the run proceeds to the first FULL epoch (3), which fails to beat
    the incumbent and fires patience; the shipped object equals the cap-2
    PUBLISHED object bit-for-bit — never the checkpoint's older epoch-1
    snapshot. (Under the legacy rule the replay itself was the strike and
    the run ended at epoch 2 — the cont2/cont3 artifact.) The rule is on
    the marker, the sidecar and the fingerprint."""
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    pool_sha = t2._sha256_file(pool)
    cfg = _driver_cfg(tmp_path, epochs=2)
    cfg["train_kwargs"]["default"]["early_stop_patience"] = 1
    bank = tmp_path / "bank"
    out = bank / "cells_all"
    ck = tmp_path / "ckpt"
    kw = dict(cfg=cfg, save_dir=out, pairs_path=str(pool), pool_sha256=pool_sha,
              config_sha="cfgsha", checkpoint_dir=ck, checkpoint_every=3)
    o1 = t2.train_si_objects(model, tok, SI, docs, [ARM], **kw)
    assert o1.arms[ARM].status == "trained", o1.arms[ARM]
    cell = out / CELL
    fr = json.loads((cell / "train_budget_meta.json").read_text())["fit_record"]
    assert fr["best_epoch"] == 2 and fr["epochs_run"] == 2, (
        "premise: the prior fit's best must be its final epoch (postdating "
        f"the mid-epoch checkpoint); got {fr['best_epoch']}/{fr['epochs_run']}")
    _seat_unbeatable_published_best(cell)
    fp_old = json.loads((cell / "COMPLETE").read_text())["fit_fingerprint"]
    assert t2.validated_complete(out, CELL, fp_old) == cell / "object.pt"
    published = _leaves(cell / "object.pt")
    ckpt = _retain(cell, ck)
    state = cont.load_state(ckpt)
    assert state["stopper"]["best_epoch"] == 1          # the OLDER best
    old_snapshot = [t.clone() for t in state["stopper"]["snapshot"]]
    assert cont._tensor_multiset(old_snapshot) != cont._tensor_multiset(published)
    snap_bank = tmp_path / "bank_r2cap24"
    sc.snapshot_cell(bank, snap_bank, CELL, superseded_by={"reason": "t"})
    o2 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=4,
                             continue_from=ckpt, **kw)
    assert o2.arms[ARM].status == "trained", o2.arms[ARM]
    marker = json.loads((cell / "COMPLETE").read_text())
    cf = marker["continued_from"]
    assert cf["incumbent"]["source"] == "published_object"
    assert cf["incumbent"]["best_epoch"] == 2
    assert cf["incumbent"]["best_heldout_ce"] == 0.0
    assert cf["incumbent"]["since_best"] == 0
    assert cf["incumbent"]["checkpoint_best_epoch"] == 1
    # the rule, on the record
    assert cf["stopper_rule"] == RULE and cf["prior_stopper_rule"] is None
    assert cf["replay_exempt_epoch"] == 1 and cf["resume_epoch"] == 1
    assert cf["prior_replayed_partial_epochs"] == []
    assert cf["replayed_partial_epochs"] == [2]
    assert cf["chain"][0]["stopper_rule"] == RULE
    # the replay was not a strike: the first FULL epoch (3) was
    assert cf["stop_reason"] == "early_stop"
    assert cf["epochs_run_total"] == 3 and cf["best_epoch"] == 2
    assert cf["epochs_from_resume_epoch_inclusive"] == 2
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    assert meta["continuation_stopper_rule"] == RULE
    assert meta["continuation_replay_exempt_epoch"] == 1
    curve = meta["fit_record"]["heldout_curve"]
    assert len(curve) == 4 and curve[2] > 0.0     # the replayed value is KEPT
    assert meta["fit_record"]["replayed_partial_epochs"] == [2]
    assert meta["fit_record"]["best_epoch"] == 2
    shipped = _leaves(cell / "object.pt")
    assert len(shipped) == len(published)
    for a, b in zip(shipped, published):
        assert torch.equal(a, b), "shipped != the cap-2 published best"
    assert cont._tensor_multiset(shipped) != cont._tensor_multiset(old_snapshot)
    assert (cell / "object.pt").read_bytes() == \
        (snap_bank / "cells_all" / CELL / "object.pt").read_bytes()
    # the fingerprint carries the rule: not the legacy marker's
    summ = cont.summarize(state, ckpt)
    fp = lambda m: t2.cell_fit_fingerprint(  # noqa: E731
        cfg, SI, ARM, positions="all", pool_sha256=pool_sha,
        config_sha="cfgsha", continued_from=m)
    assert marker["fit_fingerprint"] == fp(cont.continuation_marker(summ, 4))
    assert marker["fit_fingerprint"] != fp(
        cont.continuation_marker(summ, 4, stopper_rule=None))
    # the publication is continuable again: its retained state is consistent
    ckpt2 = _retain(cell, ck)
    st2 = cont.load_state(ckpt2)
    assert st2["replayed_partial_epochs"] == [2]
    assert cont.verify_cell(cell, st2)["problems"] == []


def test_fits_spawn_continue_cap_merge_keeps_unrelated_rows(sf, fake,
                                                            tmp_path):
    """Finding 2: a SPARSE --cells selector across two SIs x two arms
    replaces exactly the named cells in an existing manifest; the other
    cells of the same SIs/arms (possibly-running calls the watcher tracks)
    and another model's rows survive untouched."""
    cfg_local = sf.WORKTREE / Path(
        sf.MODELS["gemma4e4b_base"]["cfg"]).relative_to("/repo")
    sha = sf._config_sha16(cfg_local)

    def remote(item):
        key, pairs, cap, root = item[:4]
        cells = [f"{s}__{a}" for s, a in pairs]
        return {"model": key, "config_sha": sha, "missing_pools": [],
                "continue_cap": cap, "fingerprints": {c: FP_NEW for c in cells},
                "continuations": {c: {"prior_fingerprint": FP_OLD,
                                      "resume_step": 1200, "resume_epoch": 23}
                                  for c in cells},
                "missing_checkpoints": []}

    fake.remote = remote
    root = "out/gemma4e4b_base"

    def row(model, si, arm, cid):
        return {"model": model, "si": si, "arm": arm, "positions": "all",
                "leg": "all", "cell_id": f"{si}__{arm}",
                "out_root": root if model == "gemma4e4b_base"
                else "out/qwen35_4b_base",
                "expect_fingerprint": "c" * 64, "app": "sdfbase-fits",
                "function": "fit_h200_std", "function_id": "fu-old",
                "config_sha": sha, "overwrite": False, "checkpoint_every": 100,
                "continue_cap": 0, "call_id": cid, "spawn_time": "t",
                "status": "spawned (in-container skip expected)"}

    existing = [row("gemma4e4b_base", "constv2-mix-1m", "psr_ce_all", "fc-1"),
                row("gemma4e4b_base", "constv2-mix-1m", "lora_r4_ce", "fc-2"),
                row("gemma4e4b_base", "constv2-mix-10m", "psr_ce_all", "fc-3"),
                row("gemma4e4b_base", "constv2-mix-10m", "lora_r4_ce", "fc-4"),
                row("qwen35_4b_base", "constv2-mix-1m", "psr_ce_all", "fc-5")]
    mpath = tmp_path / "fit_calls.json"
    mpath.write_text(json.dumps({"app": "sdfbase-fits", "cells": existing}))
    sf.main(mode="fits-spawn", model="gemma4e4b_base",
            cells="constv2-mix-1m__psr_ce_all,constv2-mix-10m__lora_r4_ce",
            continue_cap=34, manifest=str(mpath))
    rows = {(r["model"], r["cell_id"]): r
            for r in json.loads(mpath.read_text())["cells"]}
    assert len(rows) == 5
    # the two named cells were replaced by continuation rows
    for cid in ("constv2-mix-1m__psr_ce_all", "constv2-mix-10m__lora_r4_ce"):
        r = rows[("gemma4e4b_base", cid)]
        assert r["continue_cap"] == 34 and r["expect_fingerprint"] == FP_NEW
        assert r["call_id"].startswith("fc-succ-")
    # the product's OTHER cells and the other model's row are untouched
    for key, cid, call in (("gemma4e4b_base", "constv2-mix-1m__lora_r4_ce", "fc-2"),
                           ("gemma4e4b_base", "constv2-mix-10m__psr_ce_all", "fc-3"),
                           ("qwen35_4b_base", "constv2-mix-1m__psr_ce_all", "fc-5")):
        r = rows[(key, cid)]
        assert r["call_id"] == call and r["continue_cap"] == 0
        assert r["expect_fingerprint"] == "c" * 64
    assert len(fake.spawned) == 2


def test_publish_cell_marker_extra_is_additive_only(tmp_path):
    """The continued_from block may only ADD to the COMPLETE marker — never
    shadow an identity field (a minimal staged temp dir is enough: the
    clash check runs before the marker is written)."""
    tmp = tmp_path / ".tmp"
    tmp.mkdir()
    (tmp / "object.pt").write_bytes(b"x")
    (tmp / "train_budget_meta.json").write_text("{}")
    (tmp / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="may not override"):
        t2.publish_cell(tmp_path, CELL, tmp, si_id=SI, arm=ARM,
                        positions="all", audit={}, fingerprint=FP_NEW,
                        marker_extra={"fit_fingerprint": FP_OLD})
    assert not (tmp_path / CELL / "COMPLETE").exists()


# --------------------------------------------------------------------------- #
# (g) chained continuation, the lr multiplier, the deployment pin              #
# --------------------------------------------------------------------------- #


def test_marker_lr_mult_and_publication_marker_round_trip():
    """1.0 leaves the marker byte-identical to the pre-multiplier marker (so
    every fingerprint stamped before the multiplier still resolves); a
    non-1.0 multiplier joins it; the marker a published continuation ran
    under is rebuilt from its legacy continued_from fields exactly."""
    summ = cont.summarize(_synthetic_state())
    legacy = {"fit_fingerprint": FP_OLD, "step": 6, "epoch": 1,
              "batches_done": 2, "epochs_cap": 4}
    # (stopper_rule=None: the legacy marker; the default adds the rule
    # — pinned in the (h) section)
    NEW = {"stopper_rule": cont.STOPPER_RULE_REPLAY_EXEMPT}
    assert cont.continuation_marker(summ, 4, stopper_rule=None) == legacy
    assert cont.continuation_marker(summ, 4, 1.0, None) == legacy
    assert cont.continuation_marker(summ, 4) == {**legacy, **NEW}
    assert cont.continuation_marker(summ, 4, lr_mult=0.25,
                                    stopper_rule=None) == {
        **legacy, "lr_mult": 0.25}
    assert cont.continuation_marker(summ, 4, lr_mult=0.25) == {
        **legacy, "lr_mult": 0.25, **NEW}
    assert cont.marker_lr_mult(legacy) == 1.0
    assert cont.marker_lr_mult({**legacy, "lr_mult": 0.25}) == 0.25
    for bad in (0, -1.0, float("inf"), float("nan"), "x", None):
        with pytest.raises(cont.ContinuationError):
            cont.continuation_marker(summ, 4, lr_mult=bad)
    # the driver's continued_from block -> the marker it was fingerprinted with
    block = {"prior_fit_fingerprint": FP_OLD, "resume_step": 6,
             "resume_epoch": 1, "resume_batches_done": 2, "new_cap": 4,
             "prior_cap": 2, "stop_reason": "cap"}
    assert cont.marker_from_publication(block) == legacy
    assert cont.marker_from_publication({**block, "lr_mult": 0.25}) == {
        **legacy, "lr_mult": 0.25}
    assert cont.marker_from_publication({**block, "lr_mult": 1.0}) == legacy
    with pytest.raises(cont.ContinuationError, match="lacks"):
        cont.marker_from_publication({k: v for k, v in block.items()
                                      if k != "resume_step"})
    # an explicit 1.0 inside a recorded chain link is malformed (it would
    # move the fingerprint of an unchanged fit)
    with pytest.raises(cont.ContinuationError, match="1.0 explicitly"):
        cont._check_marker({**legacy, "lr_mult": 1.0}, "link")


def _fake_fp_of(marker):
    """A stand-in for cell_fit_fingerprint(continued_from=marker): a
    deterministic 64-hex digest of the marker (None -> the plain identity)."""
    import hashlib

    if marker is None:
        return "p" * 64
    return hashlib.sha256(json.dumps(marker, sort_keys=True).encode()).hexdigest()


def _write_complete(cell_dir: Path, fp: str, continued_from=None):
    cell_dir.mkdir(parents=True, exist_ok=True)
    doc = {"cell_id": cell_dir.name, "fit_fingerprint": fp, "files": {}}
    if continued_from is not None:
        doc["continued_from"] = continued_from
    (cell_dir / "COMPLETE").write_text(json.dumps(doc))


def test_resolve_chain_walks_and_refuses(tmp_path):
    """The identity gate for a retained checkpoint of ANY generation: a key
    equal to the plain identity needs no chain (the COMPLETE marker is not
    even read); any other key must be reproduced from the plain identity
    through the publication's recorded chain, link by link; every break is
    a loud refusal."""
    plain = _fake_fp_of(None)
    cell = tmp_path / CELL
    # generation 0: the key IS the plain identity; no marker consulted
    st0 = _synthetic_state(key=plain)
    r0 = cont.resolve_chain(cell, st0, plain, _fake_fp_of)
    assert r0 == {"chain": [], "depth": 0, "root_fit_fingerprint": plain,
                  "prior_cap": None, "prior_lr_mult": 1.0,
                  "prior_stopper_rule": None}
    # generation 1: a publication that continued the plain fit to cap 4,
    # recorded in the LEGACY shape (no chain field — the cap-34 objects)
    m1 = {"fit_fingerprint": plain, "step": 6, "epoch": 1, "batches_done": 2,
          "epochs_cap": 4}
    c1 = _fake_fp_of(m1)
    _write_complete(cell, c1, continued_from={
        "prior_fit_fingerprint": plain, "resume_step": 6, "resume_epoch": 1,
        "resume_batches_done": 2, "new_cap": 4})
    st1 = _synthetic_state(key=c1, step=12, epoch=3, bd=0)
    r1 = cont.resolve_chain(cell, st1, plain, _fake_fp_of)
    assert r1["chain"] == [m1] and r1["depth"] == 1
    assert r1["prior_cap"] == 4 and r1["prior_lr_mult"] == 1.0
    assert r1["root_fit_fingerprint"] == plain
    # generation 2: the publication records the FULL chain (the new shape),
    # its second link carrying an lr multiplier
    m2 = {"fit_fingerprint": c1, "step": 12, "epoch": 3, "batches_done": 0,
          "epochs_cap": 6, "lr_mult": 0.25}
    c2 = _fake_fp_of(m2)
    _write_complete(cell, c2, continued_from={
        "prior_fit_fingerprint": c1, "resume_step": 12, "resume_epoch": 3,
        "resume_batches_done": 0, "new_cap": 6, "lr_mult": 0.25,
        "chain": [m1, m2]})
    st2 = _synthetic_state(key=c2, step=20, epoch=5, bd=0)
    r2 = cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    assert r2["chain"] == [m1, m2] and r2["depth"] == 2
    assert r2["prior_cap"] == 6 and r2["prior_lr_mult"] == 0.25
    # the same publication read through its LEGACY fields alone (chain
    # dropped) still walks: the one link rebuilt is m2, whose prior must be
    # c1 — but the plain identity is the root, so a one-link chain from a
    # two-deep publication is BROKEN at link 0 (the link continued c1, not
    # the plain fit): the chain field is what makes depth >= 2 provable
    _write_complete(cell, c2, continued_from={
        "prior_fit_fingerprint": c1, "resume_step": 12, "resume_epoch": 3,
        "resume_batches_done": 0, "new_cap": 6, "lr_mult": 0.25})
    with pytest.raises(cont.ContinuationError, match="broken at link 0"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    # refusals, each named
    _write_complete(cell, c2, continued_from={
        "prior_fit_fingerprint": c1, "resume_step": 12, "resume_epoch": 3,
        "resume_batches_done": 0, "new_cap": 6, "lr_mult": 0.25,
        "chain": [{**m1, "fit_fingerprint": "z" * 64}, m2]})
    with pytest.raises(cont.ContinuationError, match="broken at link 0"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    _write_complete(cell, c2, continued_from={
        "prior_fit_fingerprint": c1, "resume_step": 12, "resume_epoch": 3,
        "resume_batches_done": 0, "new_cap": 6, "lr_mult": 0.25,
        "chain": [m1, {**m2, "epochs_cap": 7}]})   # a link that was not run
    with pytest.raises(cont.ContinuationError,
                       match="does not belong to the fit the publication's "
                             "chain describes"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    # the chain is fine but the CONFIG moved (another plain identity)
    _write_complete(cell, c2, continued_from={
        "prior_fit_fingerprint": c1, "resume_step": 12, "resume_epoch": 3,
        "resume_batches_done": 0, "new_cap": 6, "lr_mult": 0.25,
        "chain": [m1, m2]})
    with pytest.raises(cont.ContinuationError, match="broken at link 0"):
        cont.resolve_chain(cell, st2, "q" * 64, _fake_fp_of)
    # a foreign key with no continuation on record
    _write_complete(cell, c2)
    with pytest.raises(cont.ContinuationError,
                       match="records no continuation"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    # a foreign key and no COMPLETE marker at all
    (cell / "COMPLETE").unlink()
    with pytest.raises(cont.ContinuationError, match="chain cannot be read"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    # malformed chain records
    _write_complete(cell, c2, continued_from={"chain": []})
    with pytest.raises(cont.ContinuationError, match="non-empty list"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    _write_complete(cell, c2, continued_from={"chain": [{"step": 1}]})
    with pytest.raises(cont.ContinuationError, match="lacks marker field"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)
    _write_complete(cell, c2, continued_from="nope")
    with pytest.raises(cont.ContinuationError, match="not a mapping"):
        cont.resolve_chain(cell, st2, plain, _fake_fp_of)


def _retain(cell: Path, ck: Path) -> Path:
    """What _fit_impl's retain_final_checkpoint does at publish: the
    transient checkpoint pair moves INTO <cell>/final_ckpt/, replacing a
    pre-existing one."""
    dest = cell / "final_ckpt"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    for f in sorted((ck / CELL).iterdir()):
        shutil.move(str(f), str(dest / f.name))
    shutil.rmtree(ck / CELL)
    return dest / "ckpt.pt"


def _lr_trace(cell: Path) -> list[tuple[int, float]]:
    """(step, lr) per optimizer step from the cell's metrics.jsonl."""
    out = []
    for ln in (cell / "metrics.jsonl").read_text().splitlines():
        row = json.loads(ln)
        for k, v in row.items():
            if k.endswith("/lr"):
                out.append((int(row.get("step", len(out))), float(v)))
    return out


def test_chained_continuation_is_bitwise_the_straight_fit(sf, tiny_llama,
                                                         tmp_path):
    """End to end through the driver: cap 2 -> continue to 3 ->
    continue AGAIN to 4 (the second continuation's checkpoint carries the
    FIRST continuation's key, not the plain identity). The chained fit
    publishes with the full two-link chain on record, under constant lr its
    object equals the straight cap-4 fit's bitwise, the app's pre-spawn
    derivation (fingerprint_cpu / snapshot) resolves the SAME identity from
    the same chain, and a broken chain is refused by the driver and by that
    derivation alike — before any GPU spawn."""
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    cfg = _driver_cfg(tmp_path, epochs=2)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg))
    loaded = t2._cfg_with_positions(t2._load_cfg(str(cfg_path)), "all")
    pool_sha, config_sha = t2._cli_fit_identity(
        SimpleNamespace(pairs=str(pool), config=str(cfg_path)), loaded)
    out = tmp_path / "bank" / "cells_all"
    ck = tmp_path / "ckpt"
    kw = dict(cfg=cfg, save_dir=out, pairs_path=str(pool), pool_sha256=pool_sha,
              config_sha=config_sha, checkpoint_dir=ck, checkpoint_every=3)
    cell = out / CELL

    # generation 0: cap 2
    o1 = t2.train_si_objects(model, tok, SI, docs, [ARM], **kw)
    assert o1.arms[ARM].status == "trained", o1.arms[ARM]
    fp0 = json.loads((cell / "COMPLETE").read_text())["fit_fingerprint"]
    ckpt = _retain(cell, ck)
    assert fp0 == sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM)

    # generation 1: cap 2 -> 3 — the app's derivation names the identity
    exp1 = sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=3,
                                    checkpoint=ckpt)
    assert exp1["chain_depth"] == 1 and exp1["prior_cap"] is None
    o2 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=3,
                             continue_from=ckpt, **kw)
    assert o2.arms[ARM].status == "trained", o2.arms[ARM]
    m1 = json.loads((cell / "COMPLETE").read_text())
    fp1 = m1["fit_fingerprint"]
    assert fp1 == exp1["fingerprint"] and fp1 != fp0
    cf1 = m1["continued_from"]
    assert cf1["chain"] == exp1["chain"] and cf1["chain_depth"] == 1
    assert cf1["root_fit_fingerprint"] == fp0
    assert cf1["prior_fit_fingerprint"] == fp0 and cf1["prior_cap"] == 2
    assert cf1["lr_mult"] == 1.0 and cf1["prior_lr_mult"] == 1.0
    assert cf1["epochs_run_total"] == 3
    ckpt = _retain(cell, ck)
    state1 = cont.load_state(ckpt)
    assert state1["checkpoint_key"] == fp1          # NOT the plain identity
    assert "continued_from" not in state1           # the trainer's own write

    # generation 2: cap 3 -> 4, CHAINED — refused by v11's one-equality gate,
    # proven here through the chain the COMPLETE marker records
    exp2 = sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=4,
                                    checkpoint=ckpt)
    assert exp2["chain_depth"] == 2 and exp2["prior_cap"] == 3
    assert exp2["chain"][0] == cf1["chain"][0]
    assert exp2["chain"][1]["fit_fingerprint"] == fp1
    assert exp2["root_fit_fingerprint"] == fp0
    # the cap must exceed the LAST link's cap, not the config's — unless the
    # lr multiplier differs from that link's (a same-cap re-issue with a
    # gentler restart is a different fit; the gate passes and the driver
    # would train — checked by fingerprint here, not run)
    with pytest.raises(SystemExit,
                       match="must exceed the prior cap 3 .*link 1"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=3,
                            continue_from=ckpt, **kw)
    with pytest.raises(SystemExit, match="below the prior cap 3"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=2,
                            continue_from=ckpt, **kw)
    # …but THIS checkpoint sits on the last group boundary of the cap-3
    # fit's final epoch (step 12 = epoch index 2, micro-batch 4/4): a
    # same-cap re-issue would train nothing, so it is refused before any
    # training — by the driver AND by the app's pre-spawn derivation (the
    # one shared helper), as are a lower cap and an equal cap with the
    # same multiplier
    assert (state1["epoch"], state1["batches_done"], state1["n_batches"]) \
        == (2, 4, 4)
    with pytest.raises(SystemExit, match="nothing would train"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=3,
                            continue_from=ckpt, continue_lr_mult=0.25, **kw)
    for cap, m, why in ((3, 0.25, "nothing would train"),
                        (2, 1.0, "below the prior cap 3"),
                        (3, 1.0, "must exceed the prior cap 3")):
        with pytest.raises(cont.ContinuationError, match=why):
            sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=cap,
                                     checkpoint=ckpt, continue_lr_mult=m)
        with pytest.raises(cont.ContinuationError, match=why):
            sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM,
                                        continue_cap=cap, checkpoint=ckpt,
                                        continue_lr_mult=m)
    assert json.loads((cell / "COMPLETE").read_text())["fit_fingerprint"] == fp1
    # a mid-epoch checkpoint at the same cap with a different multiplier
    # IS admissible (the cont2 re-issue): a synthetic mid-epoch position on
    # the same chain passes the app's gate and names a new identity
    mid = dict(state1)
    mid.update(epoch=1, batches_done=2, step=6, history=[0.1] * 6,
               heldout_curve=[1.0, 1.0])
    mid["stopper"] = {**state1["stopper"], "epochs_seen": 1, "best_epoch": 1}
    torch.save(mid, ckpt)
    exp_same = sf.expected_continuation(cfg_path, pool, SI, ARM,
                                        continue_cap=3, checkpoint=ckpt,
                                        continue_lr_mult=0.25)
    assert exp_same["prior_cap"] == 3 and exp_same["marker"]["lr_mult"] == 0.25
    assert exp_same["fingerprint"] not in (fp0, fp1)
    torch.save(state1, ckpt)
    o3 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=4,
                             continue_from=ckpt, **kw)
    assert o3.arms[ARM].status == "trained", o3.arms[ARM]
    m2 = json.loads((cell / "COMPLETE").read_text())
    fp2 = m2["fit_fingerprint"]
    assert fp2 == exp2["fingerprint"] and fp2 not in (fp0, fp1)
    cf2 = m2["continued_from"]
    assert cf2["chain"] == exp2["chain"] and cf2["chain_depth"] == 2
    assert [m["epochs_cap"] for m in cf2["chain"]] == [3, 4]
    assert [m["fit_fingerprint"] for m in cf2["chain"]] == [fp0, fp1]
    assert cf2["prior_fit_fingerprint"] == fp1
    assert cf2["root_fit_fingerprint"] == fp0 and cf2["prior_cap"] == 3
    assert cf2["lr_schedule"]["prior_total_steps"] == 3 * 4   # cap-3 horizon
    assert cf2["lr_schedule"]["new_total_steps"] == 4 * 4
    assert cf2["epochs_run_total"] == 4 and cf2["stop_reason"] == "cap"
    assert cf2["resumed_from_step"] is not None
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    assert meta["continued_from"] == cf2
    assert t2.validated_complete(out, CELL, fp2) == cell / "object.pt"
    assert t2.validated_complete(out, CELL, fp1) is None

    # the straight cap-4 fit publishes the SAME object (constant lr)
    cfg4 = _driver_cfg(tmp_path, epochs=4)
    out4 = tmp_path / "bank4" / "cells_all"
    o4 = t2.train_si_objects(model, tok, SI, docs, [ARM], cfg=cfg4,
                             save_dir=out4, pairs_path=str(pool),
                             pool_sha256=pool_sha, config_sha="cfgsha4",
                             checkpoint_dir=tmp_path / "ckpt4",
                             checkpoint_every=3)
    assert o4.arms[ARM].status == "trained", o4.arms[ARM]
    assert cont._tensor_multiset(_leaves(cell / "object.pt")) == \
        cont._tensor_multiset(_leaves(out4 / CELL / "object.pt")), \
        "chained object != straight cap-4 object"
    meta4 = json.loads((out4 / CELL / "train_budget_meta.json").read_text())
    assert meta4["fit_record"]["heldout_curve"] == \
        meta["fit_record"]["heldout_curve"]
    assert meta4["fit_record"]["best_epoch"] == meta["fit_record"]["best_epoch"]

    # a THIRD continuation would walk two links — and a BROKEN chain is
    # refused everywhere: by the app's derivation (what fingerprint_cpu and
    # snapshot call) and by the driver, before model load matters
    ckpt = _retain(cell, ck)
    assert sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=6,
                                    checkpoint=ckpt)["chain_depth"] == 3
    good = (cell / "COMPLETE").read_text()
    broken = json.loads(good)
    broken["continued_from"]["chain"][0]["fit_fingerprint"] = "z" * 64
    (cell / "COMPLETE").write_text(json.dumps(broken))
    with pytest.raises(cont.ContinuationError, match="broken at link 0"):
        sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=6,
                                    checkpoint=ckpt)
    with pytest.raises(SystemExit, match="broken at link 0"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=6,
                            continue_from=ckpt, **kw)
    # a chain whose last link was never run (the key does not follow)
    broken = json.loads(good)
    broken["continued_from"]["chain"][1]["epochs_cap"] = 5
    (cell / "COMPLETE").write_text(json.dumps(broken))
    with pytest.raises(cont.ContinuationError, match="chain describes"):
        sf.expected_fit_fingerprint(cfg_path, pool, SI, ARM, continue_cap=6,
                                    checkpoint=ckpt)
    with pytest.raises(SystemExit, match="chain describes"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=6,
                            continue_from=ckpt, **kw)
    # the config moved: the root no longer resolves to the chain's first link
    (cell / "COMPLETE").write_text(good)
    with pytest.raises(SystemExit, match="broken at link 0"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=6,
                            continue_from=ckpt,
                            **{**kw, "cfg": _driver_cfg(tmp_path, epochs=2,
                                                        lr=2e-3)})
    # nothing above touched the publication
    assert (cell / "COMPLETE").read_text() == good
    assert json.loads(good)["fit_fingerprint"] == fp2


def test_continue_lr_mult_scales_the_trajectory_exactly(tiny_llama, tmp_path):
    """The same retained checkpoint continued twice under
    warmup_cosine, once at --continue-lr-mult 1.0 and once at 0.25 — the
    0.25 run's lr is EXACTLY 0.25x the 1.0 run's at every optimizer step
    (the schedule is linear in the peak; a power-of-two multiplier makes the
    float products identical), the two publish under DIFFERENT fingerprints,
    the multiplier is on the record everywhere, and a 1.0 marker is the
    pre-multiplier marker byte for byte."""
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    pool_sha = t2._sha256_file(pool)
    cfg = _driver_cfg(tmp_path, epochs=2, lr_schedule="warmup_cosine")
    cfg["train_kwargs"][ARM]["warmup_steps"] = 2
    out = tmp_path / "bank" / "cells_all"
    ck = tmp_path / "ckpt"
    kw = dict(cfg=cfg, pairs_path=str(pool), pool_sha256=pool_sha,
              config_sha="cfgsha", checkpoint_every=3)
    o1 = t2.train_si_objects(model, tok, SI, docs, [ARM], save_dir=out,
                             checkpoint_dir=ck, **kw)
    assert o1.arms[ARM].status == "trained", o1.arms[ARM]
    ckpt = _retain(out / CELL, ck)
    # two copies of the publication, one continuation each
    runs = {}
    for m in (1.0, 0.25):
        bank = tmp_path / f"bank_m{m}"
        shutil.copytree(out.parent, bank)
        cell = bank / "cells_all" / CELL
        o = t2.train_si_objects(
            model, tok, SI, docs, [ARM], save_dir=bank / "cells_all",
            checkpoint_dir=tmp_path / f"ckpt_m{m}", epochs_cap=4,
            continue_from=cell / "final_ckpt" / "ckpt.pt",
            continue_lr_mult=m, **kw)
        assert o.arms[ARM].status == "trained", o.arms[ARM]
        marker = json.loads((cell / "COMPLETE").read_text())
        runs[m] = (marker, _lr_trace(cell))
    (mk1, lr1), (mk4, lr4) = runs[1.0], runs[0.25]
    assert mk1["fit_fingerprint"] != mk4["fit_fingerprint"]
    # the trace covers the continuation's steps (resume tail + new epochs)
    assert len(lr1) == len(lr4) >= 8
    assert [s for s, _ in lr1] == [s for s, _ in lr4]
    assert all(v4 == 0.25 * v1 for (_, v1), (_, v4) in zip(lr1, lr4)), \
        list(zip(lr1, lr4))[:4]
    assert max(v for _, v in lr1) > 0          # the trajectory is not ~0
    cf1, cf4 = mk1["continued_from"], mk4["continued_from"]
    assert cf1["lr_mult"] == 1.0 and cf4["lr_mult"] == 0.25
    assert "lr_mult" not in cf1["chain"][0]           # 1.0 never rides
    assert cf4["chain"][0]["lr_mult"] == 0.25
    rep1, rep4 = cf1["lr_schedule"], cf4["lr_schedule"]
    assert rep1["peak_lr"] == rep4["peak_lr"] == 1e-3
    assert rep1["effective_peak_lr"] == 1e-3
    assert rep4["effective_peak_lr"] == 0.25 * 1e-3
    assert rep4["lr_at_resume_new_horizon"] == \
        0.25 * rep1["lr_at_resume_new_horizon"]
    assert rep4["lr_at_resume_prior_horizon"] == \
        rep1["lr_at_resume_prior_horizon"]          # the checkpoint's own
    assert rep4["prior_lr_mult"] == 1.0
    # the same marker the app derives for m=1 equals the pre-multiplier one
    summ = cont.summarize(cont.load_state(ckpt), ckpt)
    assert cf1["chain"][0] == cont.continuation_marker(summ, 4)
    assert cf4["chain"][0] == cont.continuation_marker(summ, 4, 0.25)
    # the effective rate is on the sidecar too
    meta4 = json.loads((tmp_path / "bank_m0.25" / "cells_all" / CELL
                        / "train_budget_meta.json").read_text())
    assert meta4["continuation_lr_mult"] == 0.25
    assert meta4["continuation_effective_peak_lr"] == 0.25 * 1e-3
    meta1 = json.loads((tmp_path / "bank_m1.0" / "cells_all" / CELL
                        / "train_budget_meta.json").read_text())
    assert "continuation_lr_mult" not in meta1


def test_continuation_must_resume_and_train_before_publishing():
    """The publication postcondition: the trainer must report it resumed
    at (or after) the adopted step AND ran at least one optimizer step
    beyond it — a continuation that performed no update never publishes
    under a continuation identity."""
    cont = {"summary": {"step": 12}}
    t2._assert_continuation_resumed(
        cont, {"resumed_from_step": 12, "optimizer_steps_run": 13})
    t2._assert_continuation_resumed(
        cont, {"resumed_from_step": 15, "optimizer_steps_run": 16})
    # a SUCCESSOR that resumes its predecessor's later checkpoint on the
    # final group boundary (the predecessor trained 12 -> 16 and died before
    # publishing) runs zero steps of its own and still publishes: the
    # continuation as a whole trained (16 > the adopted 12)
    t2._assert_continuation_resumed(
        cont, {"resumed_from_step": 16, "optimizer_steps_run": 16})
    with pytest.raises(RuntimeError, match="did not resume"):
        t2._assert_continuation_resumed(
            cont, {"resumed_from_step": None, "optimizer_steps_run": 40})
    with pytest.raises(RuntimeError, match="did not resume"):
        t2._assert_continuation_resumed(
            cont, {"resumed_from_step": 11, "optimizer_steps_run": 40})
    with pytest.raises(RuntimeError, match="ran no optimizer step"):
        t2._assert_continuation_resumed(
            cont, {"resumed_from_step": 12, "optimizer_steps_run": 12})
    with pytest.raises(RuntimeError, match="ran no optimizer step"):
        t2._assert_continuation_resumed(
            cont, {"resumed_from_step": 12, "optimizer_steps_run": None})


# --------------------------------------------------------------------------- #
# (h) stopper rule replay_exempt — the replayed partial epoch is               #
#     logged, never a strike nor a new best                                    #
# --------------------------------------------------------------------------- #

RULE = cont.STOPPER_RULE_REPLAY_EXEMPT


class _Killed(RuntimeError):
    pass


def _kill_log(at_step: int):
    """A log_fn that raises once the trainer logs optimizer step
    ``at_step`` — an interruption between two checkpoints (the 24 h Modal
    ceiling's shape), after that step's checkpoint decision."""
    def log(data, step=None, **_):
        if (isinstance(data, dict) and step is not None
                and any(k.endswith("/ce_loss") for k in data)
                and int(step) >= at_step):
            raise _Killed(f"killed at step {step}")
    return log


def _recording_log(store: list):
    def log(data, step=None, **_):
        if isinstance(data, dict):
            store.append((step, dict(data)))
    return log


def test_replay_exempt_marker_and_fingerprint_differ_by_the_rule_only(tmp_path):
    """(d) The marker of a new continuation is the legacy marker plus
    ``stopper_rule``; the fingerprint moves by that field only; a
    publication's block round-trips the rule; an explicit legacy value never
    rides in a link; unknown rules refuse."""
    summ = cont.summarize(_synthetic_state())
    legacy = {"fit_fingerprint": FP_OLD, "step": 6, "epoch": 1,
              "batches_done": 2, "epochs_cap": 4}
    new = cont.continuation_marker(summ, 4)
    assert new == {**legacy, "stopper_rule": RULE}
    assert {k: v for k, v in new.items() if k != "stopper_rule"} == legacy
    assert cont.marker_stopper_rule(new) == RULE
    assert cont.marker_stopper_rule(legacy) == cont.STOPPER_RULE_LEGACY
    assert cont.check_stopper_rule(None) == cont.STOPPER_RULE_LEGACY
    assert cont.check_stopper_rule("strict") == cont.STOPPER_RULE_LEGACY
    assert cont.check_stopper_rule(RULE) == RULE
    cfg = _driver_cfg(tmp_path)

    def fp(m):
        return t2.cell_fit_fingerprint(cfg, SI, ARM, positions="all",
                                       pool_sha256="p" * 64, config_sha="c",
                                       continued_from=m)

    assert fp(new) != fp(legacy)
    assert fp(new) == fp(dict(new))
    with_m = {**legacy, "lr_mult": 0.25}
    assert fp({**with_m, "stopper_rule": RULE}) != fp(with_m)
    assert fp({**with_m, "stopper_rule": RULE}) != fp(new)
    # publication block -> the marker it was fingerprinted with
    block = {"prior_fit_fingerprint": FP_OLD, "resume_step": 6,
             "resume_epoch": 1, "resume_batches_done": 2, "new_cap": 4}
    assert cont.marker_from_publication(block) == legacy
    assert cont.marker_from_publication({**block, "stopper_rule": RULE}) == new
    assert cont.marker_from_publication(
        {**block, "stopper_rule": cont.STOPPER_RULE_LEGACY}) == legacy
    with pytest.raises(cont.ContinuationError, match="unknown continuation"):
        cont.marker_from_publication({**block, "stopper_rule": "lenient"})
    # chain links: the field is accepted; the legacy value never rides
    assert cont._check_marker(new, "link") == new
    with pytest.raises(cont.ContinuationError, match="legacy stopper rule"):
        cont._check_marker({**legacy, "stopper_rule": "strict"}, "link")
    with pytest.raises(cont.ContinuationError, match="unknown continuation"):
        cont._check_marker({**legacy, "stopper_rule": "x"}, "link")
    with pytest.raises(cont.ContinuationError, match="unknown continuation"):
        cont.continuation_marker(summ, 4, stopper_rule="x")
    # a legacy checkpoint summary records no replayed epochs
    assert summ["replayed_partial_epochs"] == []


def test_replay_exempt_admissibility_and_chain_rule(tmp_path):
    """(d) The cap rules under the new rule: a LEGACY chained continuation
    re-issued at the SAME cap and multiplier is admissible because the rule
    differs (the cont4 shape); the same request under the legacy rule, or
    once the prior link already ran under replay_exempt, refuses; a first
    continuation has no prior rule to differ from; an equal cap whose
    checkpoint sits inside the cap's final epoch refuses (the replayed
    epoch is the only one that would run, unjudged); resolve_chain reports
    the prior link's rule."""
    summ = cont.summarize(_synthetic_state())          # epoch idx 1, mb 2/4
    plain = "p" * 64
    depth0 = {"chain": [], "depth": 0, "root_fit_fingerprint": plain,
              "prior_cap": None, "prior_lr_mult": 1.0,
              "prior_stopper_rule": None}
    leg_link = {"fit_fingerprint": plain, "step": 6, "epoch": 1,
                "batches_done": 2, "epochs_cap": 4}
    depth1_legacy = {"chain": [leg_link], "depth": 1,
                     "root_fit_fingerprint": plain, "prior_cap": 4,
                     "prior_lr_mult": 1.0,
                     "prior_stopper_rule": cont.STOPPER_RULE_LEGACY}
    depth1_new = {**depth1_legacy,
                  "chain": [{**leg_link, "stopper_rule": RULE}],
                  "prior_stopper_rule": RULE}

    def adm(chain, cap, m=1.0, rule=RULE, summary=summ):
        return cont.check_cap_admissible(chain=chain, summary=summary,
                                         new_cap=cap, lr_mult=m,
                                         config_cap=4, stopper_rule=rule)

    got = adm(depth1_legacy, 4)
    assert got["prior_stopper_rule"] == cont.STOPPER_RULE_LEGACY
    assert got["stopper_rule"] == RULE and got["prior_cap"] == 4
    with pytest.raises(cont.ContinuationError,
                       match="must exceed the prior cap 4 .*link 1"):
        adm(depth1_legacy, 4, rule=None)
    with pytest.raises(cont.ContinuationError,
                       match="must exceed the prior cap 4 .*link 1"):
        adm(depth1_new, 4)
    assert adm(depth1_new, 4, m=0.25)["stopper_rule"] == RULE   # multiplier rule
    with pytest.raises(cont.ContinuationError,
                       match="must exceed the prior cap 4 .*config"):
        adm(depth0, 4)
    assert adm(depth0, 4, m=0.25)["prior_stopper_rule"] is None
    assert adm(depth0, 5)["prior_stopper_rule"] is None
    with pytest.raises(cont.ContinuationError, match="below the prior cap"):
        adm(depth1_legacy, 3)
    # inside the cap's final epoch, equal cap: refused under replay_exempt
    final = cont.summarize(_synthetic_state(epoch=3, bd=2, step=14))
    with pytest.raises(cont.ContinuationError, match="no epoch would be judged"):
        adm(depth1_legacy, 4, summary=final)
    # ...the legacy rule judged that tail, so a differing multiplier admits
    assert adm(depth1_legacy, 4, m=0.25, rule=None, summary=final)[
        "stopper_rule"] == cont.STOPPER_RULE_LEGACY
    # ...and a raised cap is never touched by that rule
    assert adm(depth1_legacy, 5, summary=final)["prior_cap"] == 4
    # the end-of-final-epoch tail rule still fires first, with its message
    end = cont.summarize(_synthetic_state(epoch=3, bd=4, step=16))
    with pytest.raises(cont.ContinuationError, match="nothing would train"):
        adm(depth1_legacy, 4, summary=end)
    # resolve_chain: a legacy link, then a replay_exempt link
    cell = tmp_path / CELL
    fp1 = _fake_fp_of(leg_link)
    link2 = {"fit_fingerprint": fp1, "step": 12, "epoch": 2,
             "batches_done": 4, "epochs_cap": 4, "lr_mult": 0.25}
    fp2 = _fake_fp_of(link2)
    link3 = {**link2, "fit_fingerprint": fp2, "stopper_rule": RULE}
    fp3 = _fake_fp_of(link3)
    _write_complete(cell, fp2, continued_from={"chain": [leg_link, link2]})
    r = cont.resolve_chain(cell, {"checkpoint_key": fp2}, plain, _fake_fp_of)
    assert (r["depth"], r["prior_lr_mult"]) == (2, 0.25)
    assert r["prior_stopper_rule"] == cont.STOPPER_RULE_LEGACY
    _write_complete(cell, fp3,
                    continued_from={"chain": [leg_link, link2, link3]})
    r = cont.resolve_chain(cell, {"checkpoint_key": fp3}, plain, _fake_fp_of)
    assert r["depth"] == 3 and r["prior_stopper_rule"] == RULE
    # a legacy publication block rebuilds its one link WITH the rule it ran
    # under; a first-generation key consults no chain
    _write_complete(cell, fp3, continued_from={
        "prior_fit_fingerprint": fp2, "resume_step": 12, "resume_epoch": 2,
        "resume_batches_done": 4, "new_cap": 4, "lr_mult": 0.25,
        "stopper_rule": RULE})
    assert cont.publication_chain(cell) == [link3]
    assert cont.resolve_chain(cell, {"checkpoint_key": plain}, plain,
                              _fake_fp_of)["prior_stopper_rule"] is None


def test_replayed_epoch_is_logged_not_struck_patience_counts_from_first_full_epoch(
        tiny_llama, tmp_path):
    """(a) + (b) at the trainer seam. The retained checkpoint sits mid-epoch
    (epoch index 1, micro-batch 2/4); an unbeatable incumbent (fill
    tensors) is seated. LEGACY rule: the replayed epoch 2 is a strike and
    patience 1 ends the run there. ``replay_exempt``: epoch 2 is logged
    (flagged replayed_partial=1), kept in the curve, NOT a strike — the
    run proceeds; the first FULL epoch (3) counts a strike as before and
    patience 1 fires; the incumbent ships; the checkpoint written inside
    epoch 3 carries the record and stays internally consistent. With one
    strike already banked (the 10M cells' checkpoints) and patience 2, the
    replay is still exempt and the first full epoch is strike two."""
    model, tok = tiny_llama
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD)
    fill = 0.222

    def edit(since_best):
        def _e(state):
            st = state["stopper"]
            st["best"] = 0.0          # nothing beats it
            st["best_epoch"] = 1
            st["since_best"] = since_best
            st["snapshot"] = [torch.full_like(t, fill)
                              for t in state["trainable"]]
        return _e

    def run(tag, *, since_best, patience, exempt):
        d = tmp_path / tag
        summ = _adopt(tmp_path / "p", d, FP_NEW, 6, edit=edit(since_best))
        assert (summ["epoch"], summ["batches_done"]) == (1, 2)
        logged: list = []
        extra = {"replay_exempt_epoch": 1} if exempt else {}
        got, diag, _ = _fit(model, tok, epochs=6, ckdir=d, key=FP_NEW,
                            patience=patience, log=_recording_log(logged),
                            **extra)
        return got, diag, logged, d

    # LEGACY: the replay is strike one -> patience 1 ends the run at epoch 2
    _, d_leg, logged_leg, _ = run("legacy", since_best=0, patience=1,
                                  exempt=False)
    assert d_leg["stopped_early"] and d_leg["epochs_run"] == 2
    assert d_leg["replayed_partial_epochs"] == []
    assert d_leg["replay_exempt_epoch"] is None
    assert not any(k.endswith("/replayed_partial")
                   for _, rec in logged_leg for k in rec)
    # (a) replay_exempt: not a strike -> a FULL epoch runs; (b) it strikes
    got, d, logged, dd = run("exempt", since_best=0, patience=1, exempt=True)
    assert d["replay_exempt_epoch"] == 1
    assert d["replayed_partial_epochs"] == [2]
    assert d["epochs_run"] == 3 and d["stopped_early"] is True
    assert d["best_epoch"] == 1 and d["best_heldout_ce"] == 0.0
    assert len(d["heldout_curve"]) == 4           # init, e1, e2 (replayed), e3
    assert d["optimizer_steps_run"] == 12
    for t in got.trainable:
        assert torch.equal(t, torch.full_like(t, fill))    # the incumbent ships
    flags = {rec["epoch"]: rec["object_ce/replayed_partial"]
             for _, rec in logged if "object_ce/replayed_partial" in rec}
    assert flags == {1: 1, 2: 0}
    hl = [rec["object_ce/heldout_loss"] for _, rec in logged
          if "object_ce/heldout_loss" in rec]
    assert hl == d["heldout_curve"][2:]           # logged AND kept in the curve
    assert d["heldout_curve"][2] > 0.0            # the replayed value is real
    ck = torch.load(dd / "ckpt.pt", weights_only=False)
    assert (ck["step"], ck["epoch"]) == (12, 2)
    assert ck["replayed_partial_epochs"] == [2]
    assert ck["stopper"]["epochs_seen"] == ck["epoch"] == 2
    assert ck["stopper"]["since_best"] == 0       # the replay spent no patience
    cont.load_state(dd / "ckpt.pt")               # internally consistent
    # a strike already banked (since_best 1), patience 2: replay exempt, the
    # first full epoch is strike two (legacy: the replay was strike two)
    _, d2, _, _ = run("banked", since_best=1, patience=2, exempt=True)
    assert d2["epochs_run"] == 3 and d2["stopped_early"]
    assert d2["replayed_partial_epochs"] == [2] and d2["best_epoch"] == 1
    _, d2l, _, _ = run("banked_legacy", since_best=1, patience=2,
                       exempt=False)
    assert d2l["epochs_run"] == 2 and d2l["stopped_early"]
    # nothing banked, patience 2: two FULL strikes (epochs 3, 4) end it
    _, d3, _, _ = run("two", since_best=0, patience=2, exempt=True)
    assert d3["epochs_run"] == 4 and d3["stopped_early"] and d3["best_epoch"] == 1
    assert d3["replayed_partial_epochs"] == [2]


def test_replay_exempt_is_bitwise_the_legacy_rule_when_the_replay_improves(
        tiny_llama, tmp_path):
    """(c) The rule touches bookkeeping only: the same adopted checkpoint
    continued under both rules to the cap (patience 99) trains the SAME
    trajectory — per-step losses, held-out curve, best epoch and params
    bitwise equal — when the replayed epoch improves on the incumbent and
    every later epoch improves further (so the best is a FULL epoch under
    both rules). The diagnostics differ only in the replay record."""
    model, tok = tiny_llama
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD)
    runs = {}
    for tag, extra in (("legacy", {}), ("exempt", {"replay_exempt_epoch": 1})):
        d = tmp_path / tag
        _adopt(tmp_path / "p", d, FP_NEW, 5)
        runs[tag] = _fit(model, tok, epochs=5, ckdir=d, key=FP_NEW,
                         patience=99, **extra)
    (g_l, d_l, h_l), (g_e, d_e, h_e) = runs["legacy"], runs["exempt"]
    curve = d_l["heldout_curve"]
    assert len(curve) == 6                       # init + 5 epochs
    assert curve[2] < curve[1] and all(b < a for a, b in zip(curve[2:], curve[3:])), (
        "premise: the replayed epoch (2) must improve on epoch 1 and every "
        f"later epoch improve further; got {curve}")
    assert d_l["replayed_partial_epochs"] == [] and d_e["replayed_partial_epochs"] == [2]
    assert d_l["replay_exempt_epoch"] is None and d_e["replay_exempt_epoch"] == 1
    assert h_l == h_e
    for k in ("epochs_run", "stopped_early", "heldout_curve", "best_epoch",
              "best_heldout_ce", "best_heldout_loss", "optimizer_steps_run",
              "resumed_from_step", "heldout_noop_ce", "tokens_loss_total",
              "steps_per_epoch"):
        assert d_l[k] == d_e[k], k
    assert d_l["best_epoch"] == 5 and d_l["stopped_early"] is False
    for a, b in zip(g_l.trainable, g_e.trainable):
        assert torch.equal(a, b)
    # the exempt run's final checkpoint stays continuable (epochs_seen ==
    # the epoch index although one epoch was never ruled on)
    st = cont.load_state(tmp_path / "exempt" / "ckpt.pt")
    assert st["stopper"]["epochs_seen"] == st["epoch"]
    assert st["replayed_partial_epochs"] == [2]


def test_replay_exempt_trainer_refusals(tiny_llama, tmp_path):
    """The exemption is a claim about the resume position and is refused
    where it cannot hold: without checkpoints, without a held-out stopper,
    a negative index, checkpoints configured but nothing resumed (a scratch
    fit must never run under it), or a resumed checkpoint that PREDATES the
    named epoch."""
    model, tok = tiny_llama
    obj = _fresh(model)
    exs = _examples(tok, 4)
    held = _examples(tok, 6)[4:]
    base = dict(lr=1e-3, epochs=2, batch_size=1, max_len=256, seed=0,
                lr_schedule="constant")
    ckk = dict(checkpoint_every=3, checkpoint_dir=tmp_path / "x",
               checkpoint_key=FP_NEW)
    with pytest.raises(ValueError, match="without checkpoints"):
        train_object_ce(model, tok, obj, exs, heldout=held,
                        early_stop=EarlyStopSpec(), replay_exempt_epoch=1,
                        **base)
    with pytest.raises(ValueError, match="without heldout"):
        train_object_ce(model, tok, obj, exs, replay_exempt_epoch=1,
                        **ckk, **base)
    with pytest.raises(ValueError, match="0-based"):
        train_object_ce(model, tok, obj, exs, heldout=held,
                        early_stop=EarlyStopSpec(), replay_exempt_epoch=-1,
                        **ckk, **base)
    assert not (tmp_path / "x").exists()          # refused before any work
    with pytest.raises(RuntimeError, match="no checkpoint was resumed"):
        _fit(model, tok, epochs=2, ckdir=tmp_path / "empty", key=FP_NEW,
             replay_exempt_epoch=0)
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD)
    _adopt(tmp_path / "p", tmp_path / "c", FP_NEW, 6)
    with pytest.raises(RuntimeError, match="predates the replayed epoch"):
        _fit(model, tok, epochs=6, ckdir=tmp_path / "c", key=FP_NEW,
             replay_exempt_epoch=3)


def test_successor_past_the_replayed_epoch_exempts_nothing_and_matches(
        tiny_llama, tmp_path):
    """(e) at the trainer seam: a continuation dies at step 10 (after its
    step-9 checkpoint, inside epoch 3 — past the replayed epoch 2); the
    successor runs the SAME kwarg (the row's item is unchanged), resumes
    at step 9, exempts nothing in its own process, inherits the replay
    record from the checkpoint, and reproduces the uninterrupted
    continuation bitwise."""
    model, tok = tiny_llama
    _fit(model, tok, epochs=2, ckdir=tmp_path / "p", key=FP_OLD)
    ref_dir, suc_dir = tmp_path / "ref", tmp_path / "suc"
    _adopt(tmp_path / "p", ref_dir, FP_NEW, 5)
    _adopt(tmp_path / "p", suc_dir, FP_NEW, 5)
    g_ref, d_ref, h_ref = _fit(model, tok, epochs=5, ckdir=ref_dir, key=FP_NEW,
                               replay_exempt_epoch=1)
    assert d_ref["replayed_partial_epochs"] == [2]
    with pytest.raises(_Killed):
        _fit(model, tok, epochs=5, ckdir=suc_dir, key=FP_NEW,
             replay_exempt_epoch=1, log=_kill_log(10))
    ck = torch.load(suc_dir / "ckpt.pt", weights_only=False)
    assert (ck["step"], ck["epoch"], ck["batches_done"]) == (9, 2, 1)
    assert ck["replayed_partial_epochs"] == [2]
    logged: list = []
    g_suc, d_suc, h_suc = _fit(model, tok, epochs=5, ckdir=suc_dir, key=FP_NEW,
                               replay_exempt_epoch=1,
                               log=_recording_log(logged))
    assert d_suc["resumed_from_step"] == 9
    assert d_suc["replayed_partial_epochs"] == [2]     # from the checkpoint
    flags = [rec["object_ce/replayed_partial"] for _, rec in logged
             if "object_ce/replayed_partial" in rec]
    assert flags == [0, 0, 0]                          # epochs 3, 4, 5
    assert h_suc == h_ref
    assert d_suc["heldout_curve"] == d_ref["heldout_curve"]
    assert d_suc["best_epoch"] == d_ref["best_epoch"]
    for a, b in zip(g_suc.trainable, g_ref.trainable):
        assert torch.equal(a, b)


def test_driver_successor_resumes_past_the_replayed_epoch(tiny_llama, tmp_path,
                                                          monkeypatch):
    """(e) through the driver — the 24 h path: attempt 1 of the
    continuation dies inside epoch 3 (after the step-9 transient
    checkpoint; the arm fails, nothing publishes); attempt 2 with the SAME
    flags (fits_watch's successor item) re-adopts nothing, resumes the
    transient checkpoint at step 9, applies no exemption of its own,
    carries the replay record from the checkpoint and publishes the object
    the uninterrupted continuation publishes, bitwise, under the same
    fingerprint."""
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    pool_sha = t2._sha256_file(pool)
    cfg = _driver_cfg(tmp_path, epochs=2)
    bank = tmp_path / "bank"
    out = bank / "cells_all"
    ck = tmp_path / "ckpt"
    kw = dict(cfg=cfg, save_dir=out, pairs_path=str(pool), pool_sha256=pool_sha,
              config_sha="cfgsha", checkpoint_dir=ck, checkpoint_every=3)
    o1 = t2.train_si_objects(model, tok, SI, docs, [ARM], **kw)
    assert o1.arms[ARM].status == "trained", o1.arms[ARM]
    cell = out / CELL
    ckpt = _retain(cell, ck)
    # the uninterrupted continuation, on a copy of the publication
    ref_bank = tmp_path / "ref"
    shutil.copytree(bank, ref_bank)
    ref_cell = ref_bank / "cells_all" / CELL
    o_ref = t2.train_si_objects(
        model, tok, SI, docs, [ARM], epochs_cap=5,
        continue_from=ref_cell / "final_ckpt" / "ckpt.pt",
        **{**kw, "save_dir": ref_bank / "cells_all",
           "checkpoint_dir": tmp_path / "ckpt_ref"})
    assert o_ref.arms[ARM].status == "trained", o_ref.arms[ARM]
    ref_marker = json.loads((ref_cell / "COMPLETE").read_text())
    ref_cf = ref_marker["continued_from"]
    assert ref_cf["adoption"]["adopted"] is True
    assert ref_cf["replayed_partial_epochs"] == [2]
    assert ref_cf["stopper_rule"] == RULE
    # attempt 1: the container dies at step 10 (after the step-9 checkpoint)
    import sisteer.methods as methods
    real = methods.train_object_ce

    def killer(*a, log_fn=None, **k):
        kill = _kill_log(10)

        def both(data, step=None, **kk):
            if log_fn is not None:
                log_fn(data, step=step, **kk)
            kill(data, step=step)
        return real(*a, log_fn=both, **k)

    monkeypatch.setattr(methods, "train_object_ce", killer)
    o2 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=5,
                             continue_from=ckpt, **kw)
    monkeypatch.undo()
    assert o2.arms[ARM].status == "failed", o2.arms[ARM]
    assert "killed at step 10" in (o2.arms[ARM].error or "")
    # nothing published; the transient checkpoint sits at step 9 with the record
    assert json.loads((cell / "COMPLETE").read_text())["fit_fingerprint"] \
        != ref_marker["fit_fingerprint"]
    transient = torch.load(ck / CELL / "ckpt.pt", weights_only=False)
    assert (transient["step"], transient["epoch"]) == (9, 2)
    assert transient["replayed_partial_epochs"] == [2]
    assert transient["checkpoint_key"] == ref_marker["fit_fingerprint"]
    # attempt 2 — the successor: the same item, no re-adoption, resumes at 9
    o3 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=5,
                             continue_from=ckpt, **kw)
    assert o3.arms[ARM].status == "trained", o3.arms[ARM]
    m = json.loads((cell / "COMPLETE").read_text())
    cf = m["continued_from"]
    assert m["fit_fingerprint"] == ref_marker["fit_fingerprint"]
    assert cf["adoption"]["adopted"] is False
    assert cf["adoption"]["transient_step"] == 9
    assert cf["resumed_from_step"] == 9
    assert cf["replayed_partial_epochs"] == [2]
    assert cf["stopper_rule"] == RULE and cf["replay_exempt_epoch"] == 1
    assert cf["epochs_run_total"] == ref_cf["epochs_run_total"] == 5
    assert cf["best_epoch"] == ref_cf["best_epoch"]
    assert cont._tensor_multiset(_leaves(cell / "object.pt")) == \
        cont._tensor_multiset(_leaves(ref_cell / "object.pt"))
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    ref_meta = json.loads((ref_cell / "train_budget_meta.json").read_text())
    assert meta["fit_record"]["replayed_partial_epochs"] == [2]
    assert meta["fit_record"]["heldout_curve"] == \
        ref_meta["fit_record"]["heldout_curve"]
    assert meta["continuation_replay_exempt_epoch"] == 1


def test_same_cap_reissue_of_a_legacy_chained_continuation(sf, tiny_llama,
                                                          tmp_path):
    """The cont4 shape: a cell whose live publication is a
    LEGACY chained continuation (its chain link carries no stopper_rule)
    is re-issued at the SAME cap and multiplier under replay_exempt.
    Admissible by the app's pre-spawn derivation (depth 2, prior rule
    strict) and by the driver; the legacy rule at that cap refuses; the
    new publication records the legacy link untouched plus its own
    replay_exempt link; a zero-remainder replay (the checkpoint sits at an
    epoch's end) is exempt too, and the first full epoch decides."""
    model, tok = tiny_llama
    docs = _docs(4)
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(docs, pool)
    cfg = _driver_cfg(tmp_path, epochs=2)
    cfg["train_kwargs"]["default"]["early_stop_patience"] = 1
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg))
    loaded = t2._cfg_with_positions(t2._load_cfg(str(cfg_path)), "all")
    pool_sha, config_sha = t2._cli_fit_identity(
        SimpleNamespace(pairs=str(pool), config=str(cfg_path)), loaded)
    out = tmp_path / "bank" / "cells_all"
    ck = tmp_path / "ckpt"
    kw = dict(cfg=cfg, save_dir=out, pairs_path=str(pool), pool_sha256=pool_sha,
              config_sha=config_sha, checkpoint_dir=ck, checkpoint_every=3)
    cell = out / CELL

    def fp_of(marker):
        return t2.cell_fit_fingerprint(cfg, SI, ARM, positions="all",
                                       pool_sha256=pool_sha,
                                       config_sha=config_sha,
                                       continued_from=marker or None)

    # generation 0: cap 2, its final-epoch best seated unbeatable
    o1 = t2.train_si_objects(model, tok, SI, docs, [ARM], **kw)
    assert o1.arms[ARM].status == "trained", o1.arms[ARM]
    fr = json.loads((cell / "train_budget_meta.json").read_text())["fit_record"]
    assert fr["best_epoch"] == 2, f"premise: best must be epoch 2, got {fr}"
    _seat_unbeatable_published_best(cell)
    published = _leaves(cell / "object.pt")
    ckpt = _retain(cell, ck)
    # generation 1: cap 6 — replay of epoch 2 exempt, epoch 3 strikes
    # (patience 1) -> stops at epoch 3; its retained checkpoint is the
    # step-12 write (epoch index 2, micro-batch 4/4: the epoch's END)
    o2 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=6,
                             continue_from=ckpt, **kw)
    assert o2.arms[ARM].status == "trained", o2.arms[ARM]
    m1 = json.loads((cell / "COMPLETE").read_text())
    cf1 = m1["continued_from"]
    assert cf1["epochs_run_total"] == 3 and cf1["replayed_partial_epochs"] == [2]
    assert cf1["stopper_rule"] == RULE and cf1["chain_depth"] == 1
    ckpt = _retain(cell, ck)
    state1 = cont.load_state(ckpt)
    assert (state1["epoch"], state1["batches_done"], state1["n_batches"]) \
        == (2, 4, 4)
    assert state1["stopper"]["since_best"] == 0     # written before epoch 3's eval
    assert state1["replayed_partial_epochs"] == [2]
    # REWRITE generation 1 into a LEGACY generation — what the live
    # cont2/cont3 cells are: the link without the rule, the publication and
    # the checkpoint re-keyed to the legacy fingerprint, no replay record
    link_legacy = {k: v for k, v in cf1["chain"][0].items()
                   if k != "stopper_rule"}
    fp1_legacy = fp_of(link_legacy)
    assert fp1_legacy != m1["fit_fingerprint"]
    m1["fit_fingerprint"] = fp1_legacy
    cf1["chain"] = [link_legacy]
    cf1.pop("stopper_rule")
    cf1["replayed_partial_epochs"] = []
    (cell / "COMPLETE").write_text(json.dumps(m1))
    state1["checkpoint_key"] = fp1_legacy
    state1.pop("replayed_partial_epochs")
    torch.save(state1, ckpt)
    state1 = cont.load_state(ckpt)
    assert cont.verify_cell(cell, state1)["problems"] == []
    # the SAME cap (6) and multiplier (1.0) under replay_exempt is admissible:
    # the rule differs from the legacy link's — app derivation...
    exp = sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=6,
                                   checkpoint=ckpt)
    assert exp["chain_depth"] == 2 and exp["prior_cap"] == 6
    assert exp["prior_stopper_rule"] == cont.STOPPER_RULE_LEGACY
    assert exp["stopper_rule"] == RULE and exp["replay_exempt_epoch"] == 2
    assert exp["chain"][0] == link_legacy
    assert exp["chain"][1]["stopper_rule"] == RULE
    assert exp["chain"][1]["fit_fingerprint"] == fp1_legacy
    assert exp["fingerprint"] not in (fp1_legacy, m1["fit_fingerprint"])
    # ...while the legacy rule at that cap and multiplier refuses
    chain = cont.resolve_chain(cell, state1, fp_of(None), fp_of)
    assert chain["prior_stopper_rule"] == cont.STOPPER_RULE_LEGACY
    with pytest.raises(cont.ContinuationError,
                       match="must exceed the prior cap 6"):
        cont.check_cap_admissible(chain=chain,
                                  summary=cont.summarize(state1, ckpt),
                                  new_cap=6, lr_mult=1.0, config_cap=2,
                                  stopper_rule=None)
    # ...and a cap BELOW it still refuses
    with pytest.raises(SystemExit, match="below the prior cap 6"):
        t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=5,
                            continue_from=ckpt, **kw)
    # through the driver: the zero-remainder replay of epoch 3 is exempt,
    # epoch 4 (the first full epoch) strikes -> stops; the incumbent ships
    o3 = t2.train_si_objects(model, tok, SI, docs, [ARM], epochs_cap=6,
                             continue_from=ckpt, **kw)
    assert o3.arms[ARM].status == "trained", o3.arms[ARM]
    m2 = json.loads((cell / "COMPLETE").read_text())
    cf2 = m2["continued_from"]
    assert m2["fit_fingerprint"] == exp["fingerprint"]
    assert cf2["chain"] == exp["chain"] and cf2["chain_depth"] == 2
    assert "stopper_rule" not in cf2["chain"][0]
    assert cf2["chain"][1]["stopper_rule"] == RULE
    assert cf2["prior_stopper_rule"] == cont.STOPPER_RULE_LEGACY
    assert cf2["stopper_rule"] == RULE
    assert cf2["prior_fit_fingerprint"] == fp1_legacy and cf2["prior_cap"] == 6
    assert cf2["replay_exempt_epoch"] == 2 and cf2["resume_batches_done"] == 4
    assert cf2["prior_replayed_partial_epochs"] == []
    assert cf2["replayed_partial_epochs"] == [3]
    assert cf2["incumbent"]["source"] == "checkpoint_snapshot"
    assert cf2["stop_reason"] == "early_stop"
    assert cf2["epochs_run_total"] == 4 and cf2["best_epoch"] == 2
    assert cf2["epochs_from_resume_epoch_inclusive"] == 2
    assert cf2["resumed_from_step"] == 12 and cf2["optimizer_steps_run"] == 16
    for a, b in zip(_leaves(cell / "object.pt"), published):
        assert torch.equal(a, b)
    meta = json.loads((cell / "train_budget_meta.json").read_text())
    assert meta["fit_record"]["replayed_partial_epochs"] == [3]
    assert len(meta["fit_record"]["heldout_curve"]) == 5
    assert t2.validated_complete(out, CELL, exp["fingerprint"]) == cell / "object.pt"
    # a THIRD same-cap same-multiplier issue is now the same fit -> refused
    ckpt = _retain(cell, ck)
    with pytest.raises(cont.ContinuationError,
                       match="must exceed the prior cap 6 .*link 2"):
        sf.expected_continuation(cfg_path, pool, SI, ARM, continue_cap=6,
                                 checkpoint=ckpt)


def test_verify_cell_excludes_replayed_epochs_from_the_published_best_check(
        tmp_path):
    """A publication whose best POSTDATES the checkpoint is seated as the
    incumbent only if its curve is self-consistent; under replay_exempt the
    curve may hold a REPLAYED value below the best — the record's
    ``replayed_partial_epochs`` excludes it from the minimum. Without the
    record the same curve is self-inconsistent, and a best_epoch that is
    itself a replayed epoch can never have been seated."""
    state = _synthetic_state(best_epoch=1)         # checkpoint best: epoch 1
    cells = tmp_path / "cells"
    fr = {"epochs_run": 3, "best_epoch": 3, "optimizer_steps_run": 12,
          "best_heldout_ce": 0.8, "heldout_curve": [1.5, 0.9, 0.5, 0.8],
          "replayed_partial_epochs": [2]}
    d = _stage_cell(cells, state, fit_record=fr)
    ver = cont.verify_cell(d, state)
    assert ver["problems"] == [], ver["problems"]
    assert ver["checks"]["incumbent"] == "published_object"
    assert ver["checks"]["incumbent_best_epoch"] == 3
    assert ver["checks"]["incumbent_since_best"] == 0
    assert ver["checks"]["published_replayed_partial_epochs"] == [2]
    d = _stage_cell(cells, state, cell="x__lora_r4_ce",
                    fit_record={**fr, "replayed_partial_epochs": []})
    ver = cont.verify_cell(d, state)
    assert any("self-inconsistent" in p for p in ver["problems"]), ver
    d = _stage_cell(cells, state, cell="y__lora_r4_ce",
                    fit_record={**fr, "best_epoch": 2, "best_heldout_ce": 0.5})
    ver = cont.verify_cell(d, state)
    assert any("self-inconsistent" in p for p in ver["problems"]), ver
    # a legacy record (no field) with the minimum at the best still seats
    d = _stage_cell(cells, state, cell="z__lora_r4_ce",
                    fit_record={"epochs_run": 3, "best_epoch": 3,
                                "optimizer_steps_run": 12,
                                "best_heldout_ce": 0.5,
                                "heldout_curve": [1.5, 0.9, 0.8, 0.5]})
    ver = cont.verify_cell(d, state)
    assert ver["problems"] == [] and ver["checks"]["incumbent_best_epoch"] == 3
    assert ver["checks"]["published_replayed_partial_epochs"] == []
