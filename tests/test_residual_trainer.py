"""``train_psr_residual`` + the calibrated recovery gate, CPU tiny model,
synthetic docs.

Pins: the stratified per-stratum holdout carve (thin-stratum rule,
manifest round trip); prefix/content masks partition the non-pad
positions; the 50/50 source weighting sums to the group loss under
length-sorted MIXED batches; distillation recovers a planted control's
function (ED → ~1 for an in-class always-open gate, ED > 0.5 for a gated
one, from ED = 0 at the zero-init start); early stop + best restore; the
NaN breaker; checkpoint round trip incl. ``expect`` refusal; the teacher
pre-check refusal; the planted-control generator's active fraction; the
numeric floor; the recovery verdict (recovered / not recovered / the
R(A, A) degenerate branch) and the calibration artifact, with the banned
word refused.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from residual_fixtures import random_psr
from sisteer.extraction import fitckpt
from sisteer.extraction.fitmon import TrainingDiverged
from sisteer.extraction.paired_write import (PSRPairedWriteObject,
                                             SharedStateWriteObject)
from sisteer.extraction.psr import PSRObject
from sisteer.extraction.residual import (
    BANNED_WORDS, VERDICT_NOT_RECOVERED, VERDICT_RECOVERED,
    VERDICT_UNRESOLVABLE, BannedWordError, CaptureDoc, TeacherPrecheckError,
    apply_capture_manifest, assert_no_banned_words, attainable_from_controls,
    bootstrap_ed, doc_records_from_sq, ed_summary, failure_sentence,
    make_cap_batches, measure_write_rms, numeric_floor, plant_control_object,
    recovery_verdict, residual_loss_terms, stratified_holdout,
    train_psr_residual, write_calibration)
from sisteer.models import num_layers

MAX_LEN = 96


# -- doc records: CPU-parked distances × device masks -------------------------


def _sq_tables(cb, layers, seed: int):
    """Per-layer [B, S] squared distances on the CPU — where the trainer
    parks them before building doc records."""
    g = torch.Generator().manual_seed(seed)
    B, S = cb.batch.input_ids.shape
    return {l: torch.rand(B, S, generator=g) for l in layers}


def _expected_records(cb, sq_s, sq_b, layers):
    out = []
    for r, d in enumerate(cb.docs):
        rec = {"doc_id": d.doc_id}
        for cls, m in (("content", cb.row_content[r].cpu()),
                       ("prefix", cb.row_prefix[r].cpu())):
            rec[cls] = {"n": int(m.sum()),
                        "sse_student": [float((sq_s[l][r] * m).sum()) for l in layers],
                        "sse_bare": [float((sq_b[l][r] * m).sum()) for l in layers]}
        out.append(rec)
    return out


def test_doc_records_from_cpu_distances_match_hand_computation(tiny_llama):
    model, tok = tiny_llama
    layers = _all_layers(model)
    cb = make_cap_batches(tok, _docs()[:4], batch_size=4, max_len=MAX_LEN,
                          device="cpu", label="t")[0]
    sq_s, sq_b = _sq_tables(cb, layers, 1), _sq_tables(cb, layers, 2)
    got = doc_records_from_sq(cb, sq_s, sq_b, layers)
    want = _expected_records(cb, sq_s, sq_b, layers)
    assert [g["doc_id"] for g in got] == [w["doc_id"] for w in want]
    for g, w in zip(got, want):
        for cls in ("content", "prefix"):
            assert g[cls] == w[cls]
    assert all(g[cls]["n"] > 0 for g in got for cls in ("content",))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="needs a second device: the trainer's distances are "
                           "CPU-parked while the batch masks live on the GPU")
def test_doc_records_accept_gpu_masks_with_cpu_distances(tiny_llama):
    """The first real-GPU calibration died in doc_records_from_sq with
    'Expected all tensors to be on the same device' — the CPU-only test
    model never exercises two devices. Masks on cuda × distances on CPU
    must give exactly the all-CPU records."""
    model, tok = tiny_llama
    layers = _all_layers(model)
    cb_gpu = make_cap_batches(tok, _docs()[:4], batch_size=4, max_len=MAX_LEN,
                              device="cuda", label="t")[0]
    assert cb_gpu.row_content.device.type == "cuda"
    sq_s, sq_b = _sq_tables(cb_gpu, layers, 1), _sq_tables(cb_gpu, layers, 2)
    assert sq_s[layers[0]].device.type == "cpu"
    got = doc_records_from_sq(cb_gpu, sq_s, sq_b, layers)
    want = _expected_records(cb_gpu, sq_s, sq_b, layers)
    for g, w in zip(got, want):
        for cls in ("content", "prefix"):
            assert g[cls] == w[cls]

_MIX = [
    ("advice", "Sys scaffold one\n\n", "A short advice doc about being kind."),
    ("stories", "", "A longer story body that runs on for quite a few more "
                    "tokens than the short one does, on purpose, so batches "
                    "are ragged."),
    ("advice", "Sys scaffold two\n\n", "Medium length advice body here."),
    ("stories", "", "Tiny tale."),
    ("stories", "", "Another story of middling length overall, with words."),
    ("advice", "Sys scaffold three\n\n", "Third advice text, calm and brief."),
]
_CTL = [
    ("stories", "", "Control story one with neutral register text inside."),
    ("stories", "", "Control story two, a little longer than the first, "
                    "with more clauses."),
    ("advice", "Ctl sys\n\n", "Control advice body about weather."),
    ("stories", "", "Control story three."),
    ("stories", "", "Control story four, with an additional sentence here."),
]


def _docs():
    docs = [CaptureDoc(doc_id=f"mix-{i}", source="mix", family=f, prompt=p,
                       text=t) for i, (f, p, t) in enumerate(_MIX)]
    docs += [CaptureDoc(doc_id=f"ctl-{i}", source="ctl", family=f, prompt=p,
                        text=t) for i, (f, p, t) in enumerate(_CTL)]
    return docs


def _all_layers(model):
    return list(range(num_layers(model)))


# -- capture set --------------------------------------------------------------


def test_stratified_holdout_per_stratum_and_thin_flag():
    docs = _docs()
    train, hold, man = stratified_holdout(docs, frac=0.1, seed=0)
    strata = man["strata"]
    assert set(strata) == {"mix/advice", "mix/stories", "ctl/advice",
                           "ctl/stories"}
    # every stratum with >= 2 docs holds out exactly its 10% (min 1) ...
    for s, m in strata.items():
        assert m["n_holdout"] == (0 if m["n"] < 2 else
                                  min(m["n"] - 1, max(1, round(0.1 * m["n"]))))
    # ... the 1-doc ctl/advice stratum holds out nothing and says so
    assert strata["ctl/advice"]["n"] == 1
    assert strata["ctl/advice"]["no_holdout"] is True
    assert "ctl/advice" in man["no_holdout_strata"]
    # small strata forced to the 1-doc minimum are flagged thin
    assert strata["mix/advice"]["thin"] is True
    assert set(d.doc_id for d in train) | set(d.doc_id for d in hold) == \
        set(d.doc_id for d in docs)
    assert not (set(d.doc_id for d in train) & set(d.doc_id for d in hold))
    # deterministic in the seed, different across seeds
    _, hold2, _ = stratified_holdout(docs, frac=0.1, seed=0)
    assert [d.doc_id for d in hold2] == [d.doc_id for d in hold]
    # manifest round trip re-derives the identical split
    t2, h2 = apply_capture_manifest(docs, json.loads(json.dumps(man)))
    assert [d.doc_id for d in t2] == [d.doc_id for d in train]
    assert [d.doc_id for d in h2] == [d.doc_id for d in hold]
    with pytest.raises(ValueError, match="does not describe"):
        apply_capture_manifest(docs[:-1], man)


def test_position_classes_partition_non_pad_positions(tiny_llama):
    model, tok = tiny_llama
    docs = _docs()
    cbs = make_cap_batches(tok, docs, batch_size=3, max_len=MAX_LEN,
                           device="cpu", label="t")
    for cb in cbs:
        valid = cb.batch.attention_mask.bool()
        assert torch.equal(cb.row_prefix | cb.row_content, valid)
        assert not (cb.row_prefix & cb.row_content).any()
        # prefix positions only on rows with a non-empty prompt
        for r, d in enumerate(cb.docs):
            n_pre = int(cb.row_prefix[r].sum())
            if d.has_prefix:
                assert n_pre == int(cb.batch.prompt_lens[r])
            else:
                assert n_pre == 0
        # the four (source, class) masks partition the valid positions
        tot = sum(cb.n_pos.values())
        assert tot == int(valid.sum())


def test_source_weights_sum_to_group_loss_under_mixed_batches(tiny_llama):
    """Per-micro-batch contributions Σ SSE/(N_group·d) add up to the group
    loss Σ_layers Σ_s w_s·mean_over_group_positions — exactly, so the
    50/50 weighting is a property of the GROUP, not of batch composition."""
    model, tok = tiny_llama
    docs = _docs()
    cbs = make_cap_batches(tok, docs, batch_size=2, max_len=MAX_LEN,
                           device="cpu", label="t")
    assert any(len({d.source for d in cb.docs}) == 2 for cb in cbs), \
        "fixture must produce a mixed-source batch under length sorting"
    layers = [0, 1]
    d_model = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    # synthetic per-position squared distances, one tensor per (batch, layer)
    sq = [{l: torch.rand(cb.batch.input_ids.shape, generator=g) for l in layers}
          for cb in cbs]
    from sisteer.extraction.residual import _group_denominators
    denoms = _group_denominators(cbs)
    total = 0.0
    for cb, s in zip(cbs, sq):
        loss, _parts = residual_loss_terms(
            s, cb, denoms, d_model=d_model,
            source_weights={"mix": 0.5, "ctl": 0.5})
        total += float(loss)
    # reference: pool positions by (source, class) over the whole group
    want = 0.0
    for src in ("mix", "ctl"):
        for cls in ("content", "prefix"):
            num = sum(float((s[l] * cb.masks[(src, cls)]).sum())
                      for cb, s in zip(cbs, sq) for l in layers)
            n = denoms[(src, cls)]
            if n:
                want += 0.5 * num / (n * d_model)
    assert math.isclose(total, want, rel_tol=1e-6)


# -- distillation --------------------------------------------------------------


def _run(model, tok, teacher, docs, *, epochs=6, lr=2e-2, patience=3, **kw):
    train, hold, _ = stratified_holdout(docs, frac=0.2, seed=0)
    student = PSRObject(model, _all_layers(model), train_direction=True)
    diag = {}
    from sisteer.extraction.distill import EarlyStopSpec
    hist = train_psr_residual(
        model, tok, student, teacher, train, hold, lr=lr, epochs=epochs,
        batch_size=2, accum_steps=2, max_len=MAX_LEN, seed=0,
        max_grad_norm=math.inf, diagnostics=diag,
        early_stop=EarlyStopSpec(patience=patience), **kw)
    return student, hist, diag


def test_distillation_recovers_an_in_class_planted_control(tiny_llama):
    """R(B ⊕ C, B) with an always-open C: the coherent teacher is +C
    exactly, C is in the student's class at init (gate = constant 1), so
    ED climbs from exactly 0 to ~1 on both source strata."""
    model, tok = tiny_llama
    B = random_psr(model,11)
    C = random_psr(model,23, constant_gate=True)
    teacher = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    student, hist, diag = _run(model, tok, teacher, _docs(), epochs=8,
                               lr=3e-2, patience=8)
    assert len(hist) == diag["optimizer_steps_run"]
    # the zero-init student IS the bare model: the pre-training held-out
    # loss equals the teacher deviation (ED = 0 by construction)
    assert diag["heldout_init"] > 0
    ed = diag["ed"]
    for s in ("mix", "ctl"):
        assert ed[s]["content"]["ed"] > 0.9, (s, ed[s]["content"])
        assert ed[s]["content"]["lo"] is not None
    assert diag["best_heldout_loss"] < 0.1 * diag["heldout_init"]
    # per-layer EDs and per-stratum rows exist
    assert set(ed["mix"]["content"]["per_layer_ed"]) == set(_all_layers(model))
    assert "mix/stories" in ed and "ctl/stories" in ed


def test_distillation_of_a_gated_control_beats_half(tiny_llama):
    model, tok = tiny_llama
    B = random_psr(model,11)
    C = random_psr(model,29, constant_gate=False)
    teacher = SharedStateWriteObject([(1.0, B), (1.0, C), (-1.0, B)])
    _s, _h, diag = _run(model, tok, teacher, _docs(), epochs=10, lr=3e-2,
                        patience=10)
    for s in ("mix", "ctl"):
        assert diag["ed"][s]["content"]["ed"] > 0.5, diag["ed"][s]["content"]


def test_early_stop_restores_best_and_records_curve(tiny_llama):
    model, tok = tiny_llama
    B = random_psr(model,11)
    teacher = PSRPairedWriteObject(B, PSRObject(model, _all_layers(model),
                                                train_direction=True))
    # a huge lr makes epochs oscillate → early stop fires and restores best
    student, hist, diag = _run(model, tok, teacher, _docs(), epochs=12,
                               lr=5.0, patience=1)
    assert diag["epochs_run"] <= 12
    assert len(diag["heldout_curve"]) == diag["epochs_run"] + 1
    assert diag["best_epoch"] >= 1
    # the restored weights reproduce the best held-out value
    assert math.isclose(diag["final_heldout_loss"], diag["best_heldout_loss"],
                        rel_tol=1e-5, abs_tol=1e-8)


def test_nan_breaker(tiny_llama):
    model, tok = tiny_llama
    B = random_psr(model,11)
    teacher = SharedStateWriteObject([(1.0, B)])
    train, hold, _ = stratified_holdout(_docs(), frac=0.2, seed=0)
    student = PSRObject(model, _all_layers(model), train_direction=True)
    student.directions[0].data[0] = float("nan")
    with pytest.raises(TrainingDiverged):
        train_psr_residual(model, tok, student, teacher, train, hold, lr=1e-2,
                           epochs=1, batch_size=2, accum_steps=1,
                           max_len=MAX_LEN)


def test_teacher_precheck_refusal_runs_before_any_step(tiny_llama):
    model, tok = tiny_llama
    B = random_psr(model,11)
    teacher = SharedStateWriteObject([(1.0, B)])
    train, hold, _ = stratified_holdout(_docs(), frac=0.2, seed=0)
    student = PSRObject(model, _all_layers(model), train_direction=True)
    before = [p.detach().clone() for p in student.trainable]
    with pytest.raises(TeacherPrecheckError, match="layer 3"):
        train_psr_residual(model, tok, student, teacher, train, hold, lr=1e-2,
                           epochs=1, batch_size=2, max_len=MAX_LEN,
                           teacher_precheck=lambda: ["layer 3: mismatch"])
    for p, b in zip(student.trainable, before):
        assert torch.equal(p, b)
    # a passing pre-check runs normally
    train_psr_residual(model, tok, student, teacher, train, hold, lr=1e-2,
                       epochs=1, batch_size=2, max_len=MAX_LEN,
                       teacher_precheck=lambda: [])


def test_teacher_capture_verifies_the_hook_contract_on_every_forward(tiny_llama):
    """The trainer never takes a distillation target on faith. A
    teacher whose ``paired_write`` is right but whose runtime ``apply()``
    doubles the write (or no-ops, or writes at the wrong layer) raises
    :class:`HookContractError` at the FIRST teacher forward — the held-out
    reference capture, before any optimizer step, with no
    ``teacher_precheck`` supplied at all — and the student is untouched; the
    unwrapped teacher trains. The z the check validated is the z the target
    came from: the check rides inside the capture, on every forward."""
    from residual_fixtures import ApplyPerturbed
    from sisteer.extraction.residual import HookContractError

    model, tok = tiny_llama
    B = random_psr(model,11)
    teacher = SharedStateWriteObject([(1.0, B)])
    train, hold, _ = stratified_holdout(_docs(), frac=0.2, seed=0)
    for how in ("double", "noop", "wrong_layer"):
        student = PSRObject(model, _all_layers(model), train_direction=True)
        before = [p.detach().clone() for p in student.trainable]
        with pytest.raises(HookContractError, match="HOOK CONTRACT") as ei:
            train_psr_residual(model, tok, student, ApplyPerturbed(teacher, how),
                               train, hold, lr=1e-2, epochs=1, batch_size=2,
                               max_len=MAX_LEN)
        assert isinstance(ei.value, TeacherPrecheckError)
        assert "held-out reference" in str(ei.value), how
        for p, b in zip(student.trainable, before):
            assert torch.equal(p, b)
    student = PSRObject(model, _all_layers(model), train_direction=True)
    hist = train_psr_residual(model, tok, student, teacher, train, hold,
                              lr=1e-2, epochs=1, batch_size=2, max_len=MAX_LEN)
    assert len(hist) >= 1


def test_trainer_refusals(tiny_llama):
    model, tok = tiny_llama
    B = random_psr(model,11)
    teacher = SharedStateWriteObject([(1.0, B)])
    train, hold, _ = stratified_holdout(_docs(), frac=0.2, seed=0)
    frozen_dir = PSRObject(model, _all_layers(model), train_direction=False)
    with pytest.raises(ValueError, match="train_direction"):
        train_psr_residual(model, tok, frozen_dir, teacher, train, hold,
                           lr=1e-2, max_len=MAX_LEN)
    student = PSRObject(model, _all_layers(model), train_direction=True)
    with pytest.raises(ValueError, match="holdout"):
        train_psr_residual(model, tok, student, teacher, train, [], lr=1e-2,
                           max_len=MAX_LEN)
    with pytest.raises(ValueError, match="no weight"):
        train_psr_residual(model, tok, student, teacher, train, hold, lr=1e-2,
                           max_len=MAX_LEN, source_weights={"mix": 1.0})
    with pytest.raises(ValueError, match="checkpoint"):
        train_psr_residual(model, tok, student, teacher, train, hold, lr=1e-2,
                           max_len=MAX_LEN, checkpoint_every=1)


def test_checkpoint_resume_bitwise_and_expect_refusal(tiny_llama, tmp_path,
                                                      capsys):
    model, tok = tiny_llama
    B = random_psr(model,11)
    teacher = SharedStateWriteObject([(1.0, B)])
    train, hold, _ = stratified_holdout(_docs(), frac=0.2, seed=0)

    class _Killed(RuntimeError):
        pass

    def run(ckpt_dir, kill_after=None, key="fp-resid"):
        student = PSRObject(model, _all_layers(model), train_direction=True)
        diag = {}
        calls = {"n": 0}
        orig = torch.optim.AdamW.step

        def counting_step(self, *a, **k):
            r = orig(self, *a, **k)
            calls["n"] += 1
            if kill_after is not None and calls["n"] >= kill_after:
                raise _Killed()
            return r
        torch.optim.AdamW.step = counting_step
        try:
            hist = train_psr_residual(
                model, tok, student, teacher, train, hold, lr=1e-2, epochs=3,
                batch_size=2, accum_steps=1, max_len=MAX_LEN, seed=0,
                max_grad_norm=0.5, diagnostics=diag, checkpoint_every=1,
                checkpoint_dir=str(ckpt_dir), checkpoint_key=key)
        finally:
            torch.optim.AdamW.step = orig
        return hist, [p.detach().clone() for p in student.trainable], diag

    ref_hist, ref_params, _ = run(tmp_path / "ref")
    ck = tmp_path / "killed"
    with pytest.raises(_Killed):
        run(ck, kill_after=4)
    assert (ck / fitckpt.LATEST_NAME).exists()
    res_hist, res_params, _ = run(ck)
    assert "RESUMING from checkpoint" in capsys.readouterr().out
    assert res_hist == ref_hist
    for a, b in zip(res_params, ref_params):
        assert torch.equal(a, b)
    # a checkpoint of another trainer's structure is refused (fresh start)
    st = torch.load(ck / fitckpt.LATEST_NAME, weights_only=False)
    st["trainer"] = "train_object_ce"
    torch.save(st, ck / fitckpt.LATEST_NAME)
    (ck / fitckpt.PREVIOUS_NAME).unlink(missing_ok=True)
    run(ck)
    assert "IGNORING it and starting fresh" in capsys.readouterr().out


# -- calibration pieces -----------------------------------------------------------


def test_plant_control_matches_active_fraction_and_write_rms(tiny_llama):
    model, tok = tiny_llama
    docs = _docs()
    cbs = make_cap_batches(tok, docs, batch_size=3, max_len=MAX_LEN,
                           device="cpu", label="t")
    B = random_psr(model,11)
    rms = measure_write_rms(model, B, cbs)
    assert all(v > 0 for v in rms.values())
    C, rep = plant_control_object(model, _all_layers(model), cbs, seed=3,
                                  write_rms=rms, active_frac=0.3)
    for l in _all_layers(model):
        assert rep[l]["active_frac_bare"] >= 0.3
        assert math.isclose(rep[l]["write_rms_bare"], rms[l], rel_tol=1e-4)
    # the planted C's realised write RMS under its own forward is close
    # to the target (its gates read post-write states at deeper layers)
    got = measure_write_rms(model, C, cbs)
    assert math.isclose(got[0], rms[0], rel_tol=1e-3)


def test_numeric_floor_shape_and_nonnegative(tiny_llama):
    model, tok = tiny_llama
    _, hold, _ = stratified_holdout(_docs(), frac=0.2, seed=0)
    fl = numeric_floor(model, tok, hold, layers=_all_layers(model),
                       batch_size=2, max_len=MAX_LEN)
    assert "mix" in fl["floor"] and "ctl" in fl["floor"]
    for g, row in fl["floor"].items():
        for cls in ("content", "prefix"):
            if row[cls]["n_positions"]:
                assert row[cls]["mse_sum_layers"] >= 0.0
    assert len(fl["per_doc"]) == len(hold)


def test_R_A_A_is_degenerate_unresolvable(tiny_llama):
    """R(A, A): the teacher IS bare — every stratum's teacher deviation is
    0, the verdict is 'unresolvable' + degenerate, never a pass or fail."""
    model, tok = tiny_llama
    A = random_psr(model,11)
    teacher = PSRPairedWriteObject(A, A)
    _s, _h, diag = _run(model, tok, teacher, _docs(), epochs=1, lr=1e-2)
    ed = diag["ed"]
    for s in ("mix", "ctl"):
        assert ed[s]["content"]["ed"] is None
        assert ed[s]["content"]["teacher_deviation"] == 0.0
    v = recovery_verdict(ed, {"mix": 0.9, "ctl": 0.9},
                         {"mix": {"content": {"mse_sum_layers": 1e-9}},
                          "ctl": {"content": {"mse_sum_layers": 1e-9}}})
    assert v["verdict"] == VERDICT_UNRESOLVABLE
    assert v["degenerate_parents_identical"] is True


def test_recovery_verdict_rules_and_sentences():
    def table(ed, lo, hi, dev=1.0):
        return {"content": {"ed": ed, "lo": lo, "hi": hi,
                            "teacher_deviation": dev}}
    floor = {"mix": {"content": {"mse_sum_layers": 0.01}},
             "ctl": {"content": {"mse_sum_layers": 0.01}}}
    attain = {"mix": 0.9, "ctl": 0.8}
    ok = recovery_verdict({"mix": table(0.85, 0.7, 0.9),
                           "ctl": table(0.75, 0.6, 0.8)}, attain, floor)
    assert ok["verdict"] == VERDICT_RECOVERED
    assert ok["p1_p3_rows_are"] == "confirmatory"
    assert ok["per_stratum"]["mix"]["threshold"] == pytest.approx(0.63)
    assert ok["per_stratum"]["ctl"]["threshold"] == pytest.approx(0.56)
    # lower bound below 0.7·attainable → not recovered, with the sentence
    no = recovery_verdict({"mix": table(0.85, 0.7, 0.9),
                           "ctl": table(0.6, 0.5, 0.7)}, attain, floor)
    assert no["verdict"] == VERDICT_NOT_RECOVERED
    assert no["p1_p3_rows_are"] == "descriptive"
    s = no["per_stratum"]["ctl"]["sentence"]
    assert "not recovered under this trainer + budget" in s
    assert "ED = 0.600 [0.500, 0.700] vs attainable 0.800" in s
    # the absolute 0.5 floor binds even when attainable is low
    low = recovery_verdict({"mix": table(0.45, 0.4, 0.5),
                            "ctl": table(0.75, 0.6, 0.8)},
                           {"mix": 0.5, "ctl": 0.8}, floor)
    assert low["per_stratum"]["mix"]["verdict"] == VERDICT_NOT_RECOVERED
    # a stratum within 3x the floor of bare is unresolvable
    un = recovery_verdict({"mix": table(0.85, 0.7, 0.9, dev=0.02),
                           "ctl": table(0.75, 0.6, 0.8)}, attain, floor)
    assert un["per_stratum"]["mix"]["verdict"] == VERDICT_UNRESOLVABLE
    assert un["verdict"] == VERDICT_NOT_RECOVERED  # not all recovered
    assert attainable_from_controls(
        [{"mix": table(0.9, 0.8, 0.95), "ctl": table(0.7, 0.6, 0.8)},
         {"mix": table(0.95, 0.9, 0.99), "ctl": table(0.9, 0.8, 0.95)}]
    ) == {"mix": 0.9, "ctl": 0.7}


def test_banned_word_never_appears(tmp_path):
    assert "inexpressible" in BANNED_WORDS
    s = failure_sentence(0.4, 0.3, 0.5, 0.9)
    assert_no_banned_words(s)
    with pytest.raises(BannedWordError):
        assert_no_banned_words("the residual is inexpressible in this class")
    with pytest.raises(BannedWordError):
        write_calibration(tmp_path / "c.json",
                          {"note": "Inexpressible as one gate"})
    write_calibration(tmp_path / "c.json", {"verdict": s})
    assert json.loads((tmp_path / "c.json").read_text())["verdict"] == s


def test_bootstrap_ed_collapses_for_one_doc_and_widens_for_many():
    one = [{"doc_id": "a", "stratum": "mix/stories", "source": "mix",
            "family": "stories",
            "content": {"n": 5, "sse_student": [1.0], "sse_bare": [4.0]},
            "prefix": {"n": 0, "sse_student": [0.0], "sse_bare": [0.0]}}]
    b = bootstrap_ed(one, "content", n_boot=50)
    assert b["ed"] == pytest.approx(0.75) and b["lo"] == b["hi"] == b["ed"]
    many = []
    for i in range(12):
        many.append({"doc_id": f"d{i}", "stratum": "mix/stories",
                     "source": "mix", "family": "stories",
                     "content": {"n": 5, "sse_student": [float(i % 3)],
                                 "sse_bare": [4.0]},
                     "prefix": {"n": 0, "sse_student": [0.0],
                                "sse_bare": [0.0]}})
    b = bootstrap_ed(many, "content", n_boot=200, seed=1)
    assert b["lo"] < b["ed"] < b["hi"]
    tab = ed_summary(many, layers=[0], n_boot=20)
    assert tab["mix"]["content"]["ed"] == pytest.approx(b["ed"])
    assert tab["mix"]["prefix"]["ed"] is None  # no prefix positions
