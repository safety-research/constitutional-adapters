"""sdf-base-steering round 2 — the INPUT CONTRACT of the analysis pipeline.

Every reader in compute_stats_r2.py resolves its inputs through this module,
so the harvest operator has ONE place to read what must be staged where. A
missing input never raises here: resolvers return ``None`` and the tables
mark the row PENDING (never fabricated).

Round-2 harvest root (``--harvest``, default ``$STAGE/sdfbase/harvest_r2``)
mirrors the round-1 harvest layout and extends it::

  <R2>/probe_v2/<model>/<cell>.jsonl (+ .meta.json)
        probe_score rows at scale 1.0. <model> ∈ ALL_MODELS: the two BASE
        lanes (MODELS: gemma4e4b_base, qwen35_4b_base — round-1 machinery,
        refits, derived objects, battery) and the two doc-CE lanes on
        CHAT checkpoints (WS3_MODELS: glm46vflash_doc, nemotron3nano4b_doc;
        probe render ``chat``, cells 1k…1M + ctl-100k only,
        NO adv lane, NO 10M/full, NO refits, NO derived objects, NO transfer
        battery, NO instruct positive control: the checkpoint IS the chat
        model; LANES below says what each lane carries and the tables say
        "not in this lane" rather than PENDING for the rest).
        cell ∈ {bare, constitution, <si>__<arm>, <derived cell id>}. Same
        schema as round 1:
        {id, coverage, source, section_id, margin_mean, margin_sum,
         lp_sum_consistent, lp_sum_violating, n_tok_consistent,
         n_tok_violating, tie}. meta.json carries probe_sha256 and scale.
        NEW cells: constv2-mix-10m__<arm>, constv2-mix-full__<arm>, the
        cap-24 refits (and their cap-34 continuations) under
        the SAME 1M cell names, the derived ids (DERIVED_CELLS below), and
        the scale-matched register controls constv2ctl-{1m,10m,full}__<arm>
        (CTL_SIS; pseudo-instructions of the released 45.8M control corpus —
        accepted by every loader, PENDING until staged). gate_reg trial
        rows, if retained: gatereg_r3_t<k>.jsonl. meta.render must equal
        the lane's render (MODEL_SPECS[<key>].render: raw for the base
        lanes, chat for the doc-CE lanes) when present.
  <R2>/probe_v2_x2/<model>/<cell>.jsonl (+ .meta.json)
        the 2x-strength pass — the landed Modal layout
        (probe_out_dirname: ``probe_v2`` + ``_x2``); meta.scale must be 2.0.
  <R2>/reports/<model>/report_<si>__<arm>_all.json
        fit reports (round-1 schema): arms.<arm>.detail carries
        best_epoch, epochs_run, stopped_early, early_stop.patience,
        heldout_ce_active/noop, ce_ratio, heldout_curve, gate_liveness
        (PSR: {layer: live_frac}), optimizer_steps_run, steps_per_epoch.
        ``stopped_early`` is the authoritative cap-hit signal. A CONTINUED
        fit (cap 24 -> 34 resumed from the retained checkpoint)
        additionally carries detail.continued_from = {prior_fit_fingerprint,
        prior_object_sha256, prior_cap, prior_epochs_run, prior_best_epoch,
        new_cap, resume_step, resume_epoch, resumed_from_step, stop_reason,
        incumbent: {source, best_epoch, best_heldout_ce}, lr_schedule:
        {peak_lr, lr_at_resume_prior_horizon, lr_at_resume_new_horizon,
        lr_jump_fraction_of_peak, new_total_steps, new_warmup_steps, ...},
        identity_checks: {object_sha256 (the PRIOR object), superseded_by:
        {fit_fingerprint, snapshot}, ...}, snapshot}. The effective cap of a
        continued fit is continued_from.new_cap (the † rule reads
        epochs_run == new_cap with stopped_early false), never the config's.
  <R2>/snapshots/<model>_r2cap24/cells_all/<cell>/{COMPLETE,
        train_budget_meta.json, snapshot_of.json, object.pt[.provenance.json],
        metrics.jsonl, final_ckpt/}  +  <R2>/snapshots/<model>_r2cap24/
        reports/report_<cell>_all.json
        the cap-24 publication of every continued cell, snapshotted BEFORE
        the continuation overwrote it (snapshot_cells.py --bank-suffix
        _r2cap24). COMPLETE carries {object_sha256, fit_fingerprint, files:
        {object.pt: sha, ...}, published_at}; snapshot_of.json carries the
        same shas + the source/snapshot banks; train_budget_meta.json
        carries fit_record (the cap-24 detail) and object_sha256. The
        continuation table pairs each snapshot with the live report of the
        same cell: live continued_from.prior_fit_fingerprint /
        prior_object_sha256 MUST equal the snapshot COMPLETE's, else REFUSED.
        Whether the continuation produced a NEW object is decided from
        object identity, never from report bytes: the r2-label MMLU
        identity.object_sha256 must equal the cap-24 sha (MMLU r2 scored the
        cap-24 object) and, when an r3-label MMLU run exists, its
        identity.object_sha256 decides changed (!= cap-24 sha) vs unchanged
        (== cap-24 sha); absent r3, the trainer's restore contract
        (best_epoch/best CE equal to the incumbent's -> the incumbent
        shipped) is reported as inferred, not verified.
  <R2>/cells_all/<model>/<cell>/{train_budget_meta.json, COMPLETE}
        the LIVE cell's own sidecar (the volume's out/<key>/cells_all/<cell>/), staged
        for SIDECAR_SIS × ARMS: fit_record.heldout_ce_active / heldout_ce_noop,
        heldout_pool_sha, object_sha256 — the identity every probe row of that cell must
        stamp, and the CE register share's numerator/denominator. COMPLETE cross-checks
        the sha; the fit report's detail must agree on the held-out CE.
  <R2>/probe_v3/<model>/<cell>.jsonl (+ .meta.json)
        the genre-matched probe v3g rows (probe_v3/README.md): sidecar names
        PROBE_V3_SHA_FROZEN, rows carry variant R / C (never pooled), cells PROBE_V3_CELLS
        + the full-scale derived ids; read per variant with the live-object stamp asserted.
  <R2>/mmlu/<model>/<label>/mmlu_<label>_<cell>.json
                          + mmlu_<label>_<cell>.per_question.jsonl
        scripts/sdfbase/base_mmlu.py's naming (the label in the dir AND the
        filename); a bare mmlu_<cell>.* twin is accepted only when it is the
        same record. label ∈ {r1, r2, r3, r2_x2, r5 — fresh-label evals} (--run-label / --out-dir; the
        round-1 snapshot bank scores under r1, the wave-A objects (cap-24
        refits, new cells) under r2, the post-continuation cap-34 objects
        under r3 (none + the changed cells only), the 2x pass under
        r2_x2). Each label pairs against its OWN none (same ordered_qid_hash
        rule); r3 is presented as the continuation generation. The per_question file is ONE header line
        {kind: "mmlu_per_question_header", cell, n_rows, identity} followed
        by rows {qidx, qid, subject, picked, answer, correct}; the aggregate
        json is {acc, correct, n, ci95, delta_vs_bare, cell, identity,
        per_question_file}. EVERYTHING about the run is nested under
        ``identity`` — never at the top level: {run_label, model,
        model_revision, config, config_sha256, dataset: "cais/mmlu",
        dataset_split, dataset_revision, n, requested_n, seed, shots, scale,
        ordered_qid_hash (sha256 over the qids in draw order, full hex),
        render, chat_template_bypassed, defenses_root, cell, object_path,
        object_sha256}. The header's identity is the pairing key
        (compute_stats_r2.MMLU_PAIR_KEYS: model, model_revision, dataset,
        dataset_split, dataset_revision, ordered_qid_hash, n, seed, shots,
        render, chat_template_bypassed); the aggregate's identity must equal
        it, identity.ordered_qid_hash must equal the hash recomputed from the
        rows, identity.cell / run_label must match the path. Fallback: the
        flat round-1 layout <R2>/mmlu/<model>/mmlu_<cell>.json (label
        "unlabeled"; per-question absent -> McNemar PENDING).
  <R2>/ce_score/<model>/<cell>__<pool>__scale<s>.json
        standalone held-out CE scorer output, s ∈ {1, 2}:
        {ce_active, ce_noop, ratio, scale, object_sha256, pairs_sha256,
         pool, docs: [{doc_id, n_target_tokens, nll_active_sum,
         nll_noop_sum}]}. pool ∈ CE_POOLS below. cell "bare" is not
         needed (nll_noop_sum IS the bare model).
  <R2>/derived/<kind>/<model>/<name>/
        derived_meta.json — the PRODUCER sidecar (scripts/sdfbase/
          psr_residual_distill.py compose-v / distill, lora_subtract.py):
          {kind, derived_id, derivation, model_key, model_id,
           model_revision, positions, response_header_len,
           [target_layers, template_hash, application_order],
           scale_matched, comparison,
           parent_pos: {cell_id, model_key, object_path, object_sha256,
                        budget_meta_sha256, fit_fingerprint, model_id,
                        model_revision, model_revision_source, ...,
                        data_scale},          # the corpus / mix parent
           parent_neg: {the same keys},        # the control parent
           built_at, builder, object_sha256, layers, [rank]}.
          kind is "psr_paired_write" (V), "lora_subtraction" (LoRA_sub) or
          "psr" (P_resid: a plain psr blob whose sidecar adds
          teacher: {kind: "psr_paired_write", derived_id, object_path,
          object_sha256, application_order}, capture_manifest_sha256,
          trainer, calibration, recovery_verdict, gate_liveness). No
          producer writes a "parents" key; a sidecar without parent_pos +
          parent_neg is REFUSED, never read through a fallback.
        calibration.json (P_resid only; CALIBRATION_SCHEMA below — the
          producer's file ALSO carries lr_mult, lr_mult_sweep: {budget,
          complete, lr_mults, selected_lr_mult, selection_rule, table:
          {"<mult>": {attainable_ed_content: {mix, ctl}, controls:
          {R_mix_empty, R_mixC_mix: {best_epoch, epochs_run, stopped_early,
          stop_reason, ...}}, lr, lr_mult, selected}}}, production_ed,
          production_stop_reason, recovery, sentences, budget,
          n_holdout_docs, t_arith_rms),
        calibration_controls.json (P_resid; the selected multiplier's
          planted-control run: sweep.{selected_lr_mult, table, per_mult},
          controls.{R_mix_empty, R_mixC_mix}.{best_epoch, epochs_run,
          epochs_cap, stopped_early, stop_reason, ed}, teacher_identity_check:
          {passed, teacher_hook_contract_passed, report}),
        calibration_identity_report.json / precheck_identity_report.json
          (P_resid; {passed, problems, teacher_hook_contract: {passed, ...},
          stage}) — the teacher-identity + hook-contract pass flags,
        gate_liveness.json ({layer: live_frac}, optional, V and P_resid),
        gate_liveness_detail.json ({min_live_frac, mean_live_frac,
          definition, n_probe_docs, object_sha256, ...}, optional),
        report.json (P_resid trainer report: best_epoch, epochs_run,
          epochs_cap, stopped_early, stop_reason, ...; optional).
  <R2>/hpo/trials/r3_gemma4e4b_base_psr_ce_all_all.json
        the gate_reg design (emit_gatereg_trials: trials[i].gate_reg).
  <hpo_dir>/ws3_winners_fold.json  +  <hpo_dir>/results/r1/<key>/<arm>_all/
        {r1+2_fit.json | r1_fit.json}    (Layout.hpo_dir; default
        $STAGE/sdfbase/hpo — the doc-CE HPO lives outside the harvest)
        the doc-CE lanes' HPO winners (lr / warmup_frac / max_grad_norm per arm)
        and the fit summaries they came from (winner block when the
        quadratic fit was positive-definite, else best_raw_trial; excluded
        trials with reasons). Reported as provenance in the doc-CE sections.
  <R2>/hpo/results/r3/gemma4e4b_base/psr_ce_all_all/r3_t<k>/summary.json
        per-trial tune_arm summary (best_heldout_loss, floor_ratio,
        best_epoch, epochs_run, stopped_early) PLUS the round-2 telemetry
        field gate_liveness ({layer: live_frac}) — or a sibling
        gate_liveness.json. Absent telemetry -> liveness PENDING.
  <R2>/transfer_r2/<model>/two_si_<attack>_aggregate.json
  <R2>/transfer_r3/<model>/two_si_<attack>_aggregate.json
        the r2-namespace battery aggregates (same schema as round 1's
        <repo>/out/sdfbase_transfer_<model>/two_si_<attack>_aggregate.json),
        cells: none, constv2ctl-100k__psr_ce_all, constv2ctl-100k__lora_r4_ce,
        none_r1 (round-1 bare RESPONSES regraded in the r2 judge pass), plus
        any gate-conditional extras. The harvest root is the ONLY source of
        r2 rows: the pull script stages them here from the battery's out dir
        (<repo>/out/sdfbase_transfer_<model>_r2/) and the analysis never
        reads that out dir itself (the former fallback let a harvest
        WITHOUT transfer_r2/ -- the round-1 fixture -- silently resolve
        against the live battery). Absent -> the battery is PENDING, naming
        this path.
        The r3 namespace (XFER_NAMESPACES["r3"]; staged at
        <R2>/transfer_r3/<slug>/… with EXACTLY transfer_r2/'s layout)
        carries a fresh bare anchor `none`, ALL FOUR current
        1M objects (constv2-mix-1m__{psr_ce_all,lora_r4_ce} for both models
        — the `_r2` wave's 1M rows served the round-1 objects) and `none_r2`
        (the `_r2` bare RESPONSES regraded in the r3 pass); its drift
        manifests name none / none_r2 / none_r2_original, where the original
        grades are the r2 aggregate's `none` cell. Contrasts are within-r3
        only. Absent -> PENDING naming the path.
        The it namespace (XFER_NAMESPACES["it"]; <R2>/transfer_it/<slug>/…,
        same layout) carries a fresh bare anchor,
        the INSTRUCT-DIRECT full-corpus objects (trained ON the instruct
        checkpoint under the keys <base>_it_doc, cell ids constv2-mix-full__
        <arm> — the same ids as the base lanes' mix-full cells, a different
        object: xfer_object_owner binds them to the it_doc key), gemma's
        re-run 1M LoRA on its current cap-60 object, and `none_r3` (the r3
        bare RESPONSES regraded in the it pass; original grades = the r3
        aggregate's `none`). Contrasts within-it only. Absent -> PENDING
        naming the path, never REFUSED.
        Object generation of a battery row: the aggregate records no object
        identity, but <ns dir>/controls/<cell>/controls_summary.json carries
        provenance.artifact_sha256 (the capability leg's object); the
        analysis compares it against the known generations of that cell
        (round-1 = r1-label MMLU identity sha, cap-24 = snapshot COMPLETE
        sha, cap-34 = r3-label MMLU identity sha) and labels the row; the
        attack legs of the same wave are labelled "same wave, identity not
        recorded" — never assumed.
        Per-item capability rows: <R2>/transfer_r2/<model>/controls/cells/
        <tok>__controls__*/rows.jsonl (staged alongside; absent -> unpaired).
        XSTest per-prompt labels: <xfer_private>/sdfbase_transfer_<model>_r2/
        {live_two_si_grade_cache*.jsonl, <cell>/xstest_responses.jsonl}
        (Layout.xfer_private; default $STAGE/sdfbase/xfer_private).
        Either absent -> that contrast is reported unpaired.
  <R2>/continuation_events.json   (optional)
        [{cell, model, step, epoch, cause, successor_call}] for the
        trainer audit.
  <R2>/refits.json                 (REQUIRED once any refit is harvested)
        {"<model>": ["constv2-mix-1m__psr_ce_all", ...]} — the cells the
        operator DECLARES as this round's scratch refits (from the wave-A
        manifest rows). A declared cell whose files differ from round 1 is
        the refit (ghost row + before/after); a declared cell whose files
        are byte-identical to round 1 is "refit not harvested"; an
        UNDECLARED cell whose files differ from round 1 is an integrity
        alarm and a --check-r1 failure. Refit status is tracked PER CELL.
  <transfer_r2 dir>/drift_manifests/<attack>.json   (drift inset)
        DRIFT_MANIFEST_SCHEMA below: ordered {goal_id, response_sha256}
        rows for the r2 bare cell, the regraded round-1 bare cell and the
        round-1 original grading, plus judge/pass provenance. The judge-drift
        leg asserts byte-identical responses (same ordered rows) between the
        original and regraded round-1 cells; the generation-drift leg asserts
        identical judge config + pass id between the two r2-pass cells. A
        missing manifest -> PENDING; any mismatch -> REFUSED with goal ids.

Validation the loaders enforce — violations are fatal (probe provenance,
bare row) or filed as REFUSED with the offending ids:
  * every probe .jsonl MUST have its .meta.json; meta.probe_sha256 MUST equal
    the frozen PROBE_SHA_FROZEN; a scale-1 file claiming another scale is
    fatal; an x2 file without meta.scale is PENDING (never defaulted).
  * sidecar IDENTITY must agree with the path: meta.model / meta.revision ==
    MODEL_SPECS[<model key>] (cross-checked against configs/sdfbase_<key>_fits.yaml);
    meta.cell == the file stem (null on bare/constitution, whose meta.context
    names the row and whose meta.object is null); meta.object ends with
    …/<model key>/…/<cell>/object.pt (derived rows: the derived dir; gate_reg
    rows: the trial token). Bare contradiction fatal; other rows REFUSED; a
    contradiction inside the frozen round-1 harvest fatal.
  * ce_score pool registry: seeded ONLY from the authoritative manifest named
    by CE_POOL_MANIFEST (pairs_<stem>.manifest.json: doc_ids_in_order,
    pool_sha256, n_docs). Absent -> that pool's comparisons PENDING;
    malformed -> REFUSED; a score file never defines the doc set or sha.
  * every probe file carries exactly 377 unique in-coverage + 76 unique
    out-of-coverage ids; every paired comparison requires the SAME id set.
  * ce_score: embedded cell/pool/scale must match the filename; the
    aggregate is recomputed from docs and must agree; doc ids unique; all
    files scoring one pool must cite the same pairs_sha256 and the same doc
    set (the pool manifest's doc_ids_in_order when one exists); a derived
    cell's object_sha256 must equal its derived_meta.json object_sha256.
  * derived_meta.json parent_pos / parent_neg must name the mix / ctl cells
    the decomposition uses, and kind / derived_id / model_key must name this
    object (P_resid's teacher must be the staged V); the parents'
    object_sha256 + fit_fingerprint are carried through, and a parent's
    ce_score object_sha256 / MMLU identity.object_sha256 must equal them
    (the parents scored are the parents composed); probe meta.object for a
    derived cell must point at that cell's dir.
  * MMLU pairing (McNemar contrasts, dose_2x) reads identity.* ONLY: both
    sides must carry and agree on MMLU_PAIR_KEYS (ordered_qid_hash first);
    a missing key is PENDING naming the key and the file, a difference is
    REFUSED. r1 pairs with r1 and r2 with r2; the one cross-label pair is a
    cell's r2_x2 run against its r2 run (identity.scale 2.0 vs 1.0).

Round-1 sources (read-only, never modified): ``--harvest-r1`` (default
$STAGE/sdfbase/harvest), round-1 derived_stats.json /
transfer_stats.json under $STAGE/sdfbase/analysis/, and the round-1
transfer aggregates under <repo>/out/sdfbase_transfer_<model>/ (for the
judge-drift leg of the drift inset).
"""
from __future__ import annotations

import hashlib
import re
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HOME = Path.home()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from site_paths import main_checkout, site_path  # noqa: E402
# the campaign's staging directory ($STAGE) and its worktree of the main checkout
STAGE = site_path("$STAGE/sdfbase", "STAGE")
POOLS_DIR = STAGE / "pools"
PROBE_SHA_FROZEN = "5c69bf41e10247c8b8640a5d78e325a5a09a023ecd8ed283c90ec98a00578eb7"
PROBE_N = {"in": 377, "out": 76}
R1_HARVEST = STAGE / "harvest"
R1_ANALYSIS = STAGE / "analysis"
R1_DERIVED = R1_ANALYSIS / "derived_stats.json"
R1_TRANSFER = R1_ANALYSIS / "transfer_stats.json"
R1_POOLS = STAGE / "pools/pools_summary.json"
R1_FILED = STAGE / "probe_scores_v2"
REPO = main_checkout() / "worktrees" / "sdf-base-steering"
XFER_PRIVATE = STAGE / "xfer_private"
R2_HARVEST_DEFAULT = STAGE / "harvest_r2"

MODELS = ["gemma4e4b_base", "qwen35_4b_base"]              # BASE lanes (round-1 machinery)
WS3_MODELS = ["glm46vflash_doc", "nemotron3nano4b_doc"]     # doc-CE lanes on CHAT checkpoints
ALL_MODELS = MODELS + WS3_MODELS
ARMS = ["psr_ce_all", "lora_r4_ce"]
MIX_SIS_R1 = ["constv2-mix-1k", "constv2-mix-10k", "constv2-mix-100k",
              "constv2-mix-1m"]
MIX_SIS_NEW = ["constv2-mix-10m", "constv2-mix-full"]
MIX_SIS = MIX_SIS_R1 + MIX_SIS_NEW
ADV_SIS = ["constv2-adv-1k", "constv2-adv-100k"]
CTL_SI = "constv2ctl-100k"
#: every decomposition table row and figure caption carries this until the
#: matched control cells are refit on the value-scanned r2 control corpus. The 100k control cell
#: (38 documents) is drawn from the ROUND-1 control corpus (control-20260901, 159k tokens).
PROVISIONAL_CONTROL_LABEL = ("PROVISIONAL: control corpus = the round-1 159k-token control "
                             "(control-20260901; the 100k control cell is its draw) — every subtraction, "
                             "derived object and verdict is re-derived once the value-scanned r2 control "
                             "corpus is trained on")
# scale-matched register controls (pseudo-instructions of the released 45.8M
# control corpus, control-scale-20260903): accepted by every loader, PENDING
# until a fits lane stages them; MATCHED_CTL names the control each mixed
# cell is budget-matched against
#: the r2 register-matched control at the mix-full budget — a NEW pseudo-instruction prefix
#: (`constv2ctl2-`; the r2 corpus re-mints production's id grammar, so `constv2ctl-` is never reused
#: for r2 pools), budget-matched to mix-full BY CONSTRUCTION (a nested seeded prefix of exactly
#: mix-full's official token count, read from mix-full's pool manifest), cap 16 / patience 2.
#: It SUPERSEDES the `constv2ctl-full` slot, which no lane will ever produce under that name;
#: `constv2ctl-1m` / `constv2ctl-10m` stay as slots (only the full scale is in scope, so nothing
#: produces them either — reported PENDING, never guessed).
CTL2_SI = "constv2ctl2-full"
CTL2_TOKENS = 42_785_435
CTL2_EPOCH_CAP, CTL2_PATIENCE = 16, 2
CTL_SIS_NEW = ["constv2ctl-1m", "constv2ctl-10m", CTL2_SI]
CTL_SIS = [CTL_SI] + CTL_SIS_NEW
MATCHED_CTL = {"constv2-mix-100k": CTL_SI, "constv2-mix-1m": "constv2ctl-1m",
               "constv2-mix-10m": "constv2ctl-10m", "constv2-mix-full": CTL2_SI}
#: the label a decomposition / verdict row carries for the CONTROL it subtracts:
#: rows on the round-1 159k control keep the PROVISIONAL tag; rows on the r2 control read "r2 control"
R2_CONTROL_LABEL = ("r2 control: register-matched, value-scanned control corpus control-scale-20260903-r2 "
                    f"(R2-D48), budget-matched to mix-full's {CTL2_TOKENS:,} official tokens, cap {CTL2_EPOCH_CAP} / "
                    f"patience {CTL2_PATIENCE} — the label the 100k rows' PROVISIONAL tag is lifted to. Provenance: the "
                    "ctl2 cells EARLY-STOP on the shared constitutional held-out pool (constv2-heldout, the fit's own "
                    "early-stop pool, like every other cell); the control's OWN held-out CE (constv2ctl2-heldout, 128 docs / "
                    "318,137 tokens, pool sha 39e2452a…) is reported post hoc and is the P3 pool for full − full (R2-D57). The r2 "
                    "control corpus is ACCEPTED as-is (user, R2-D67): its two new stock story beats ('screwed to the wall' 1.26%, "
                    "'the wall beside the' 1.29% of stories; ≤~1.3% leakage) remain as a ONE-SIDED residual on every full − full "
                    "decomposition number")
#: what the derived object and r ARE, in plain words — rendered once at the head of the decomposition tables
#: and as the caption lead of residual_verdicts.png / decomposition_matched.png
D1_PLAIN_WORDS = (
    "Every m is a probe margin (mean per-token logprob(consistent) − logprob(violating)) on the same frozen 377 "
    "in-coverage items, paired item by item. m_obj − m_bare is the derived object's shift. m_mix − m_ctl is the "
    "difference between its two PARENTS' shifts — the corpus-trained and control-trained objects at the same 100k "
    "budget, the very objects V, P_resid and LoRA-sub were built from — i.e. the values-specific shift (register "
    "cancels). D1 = (m_obj − m_bare) − (m_mix − m_ctl) per item: what the derived object shifted minus what a "
    "values-only object should have; mean D1 ≈ 0 ⇒ it carries the values part. r = mean(D1)/mean(m_ctl − m_bare) "
    "divides that leftover by the control object's own (register) shift: ≈0 removed, <0 over-removed, ≈1 untouched. "
    "1M rows pair mix-1M with ctl-100k (budget-mismatched) and stay descriptive until the r2 control objects exist.")
CONTROL_LABEL = {CTL_SI: PROVISIONAL_CONTROL_LABEL, CTL2_SI: R2_CONTROL_LABEL}


def control_label(ctl_si: str) -> str:
    return CONTROL_LABEL.get(ctl_si, f"scale-matched control {ctl_si}: no lane produces this id (PENDING slot)")
ALL_SIS = MIX_SIS + ADV_SIS + CTL_SIS
REFIT_SI = "constv2-mix-1m"          # the four cap-24 scratch refits
# the doc lanes' design: 1k…1M mixed + the 100k register control, nothing else
WS3_SIS = MIX_SIS_R1 + [CTL_SI]
SNAPSHOT_SUFFIX = "_r2cap24"         # snapshot_cells.py --bank-suffix — the FIRST bank; see SNAPSHOT_BANK_RE
#: a continuation marker names its pre-continuation snapshot as /vol/out/<model_key>_r2cap<N>/cells_all/<cell>
#: (cap 24 -> 34; cap 8 -> 32 and cap 4 -> 16 land in _r2cap8 / _r2cap4). The analysis resolves
#: the bank FROM THE MARKER — never from a fixed suffix — under <harvest>/snapshots/<model_key>_r2cap<N>/.
#: bank suffixes carry the prior cap and an optional letter variant for a re-issue of the same cap
#: (_r2cap34b for the chained gemma 1M LoRA, _r2cap8b / _r2cap4b for the quarter-peak re-issues)
SNAPSHOT_BANK_RE = re.compile(r"^(?:.*/)?(?P<model>[^/]+?)(?P<suffix>_r2cap(?P<cap>\d+)(?P<variant>[a-z]?))/cells_all/(?P<cell>[^/]+)/?$")


def bank_cap(suffix: str) -> int:
    """The prior cap a bank suffix names: '_r2cap8b' -> 8."""
    m = re.match(r"^_r2cap(\d+)[a-z]?$", suffix or "")
    if not m:
        raise ValueError(f"not a snapshot bank suffix: {suffix!r}")
    return int(m[1])


def parse_snapshot_marker(marker) -> Optional[dict]:
    """{model, suffix, cap, cell} from a continued_from.snapshot path, or None when the path is not
    a snapshot-bank cell path."""
    m = SNAPSHOT_BANK_RE.match(str(marker or ""))
    return ({"model": m["model"], "suffix": m["suffix"], "cap": int(m["cap"]), "variant": m["variant"] or None,
             "cell": m["cell"]} if m else None)

#: what the +constitution in-context condition actually is, stated wherever it is
#: interpreted (scaling reference line, probe validity, the doc-lane provisional note)
IN_CONTEXT_NOTE = (
    "the in-context text is the 8-section, 4,896-word SI digest of the constitution (6,027 gemma tokens), not the "
    "constitution itself; 64% of the sectioned probe items were built from constitution sections the digest does NOT "
    "contain, so the in-context positive control could not 'see' the relevant section for most sectioned items "
    "(R2-D52). Render: gemma-4 has a NATIVE system block — one <bos>, then the system turn, the user turn and the "
    "model turn (the SI sits in that native system block, not in the user turn). The round-1 D20 raw-render gemma-it rows were BOS-less "
    "(the gemma-4-E4B-it post-processor adds no <bos> template-free): they sit at ≈ −10 nats/tok with control shifts at "
    "noise and one prepended <bos> restores them — a MISSING BOS, not a model fact; the chat-render rows are BOS-correct "
    "(R2-D61; R2-D43 + R2-D52). gemma-it's in-context null on the full-passage mean hides a decision-point effect "
    "(+0.12 [+0.06, +0.19] on the first sentence / first 16 tokens / diverging span of the ORIGINAL pairs, R2-D61) — "
    "which the genre-matched probe v3 shows to be REGISTER (refusal-shaped openers), not values: within a genre "
    "gemma-it has no likelihood tilt toward constitution-consistent content at any span (R2-D63)")
GEMMA_D21_NOTE = ("gemma's in-context positive control: null on the full-passage mean; the decision-span positive on the "
                  "original pairs is a register effect (refusal-shaped openers) — genre-matched pairs show no within-genre "
                  "tilt toward constitution-consistent content at any span, so a likelihood readout of constitutional VALUES "
                  "on gemma is ≈ nil while its behavioural response is large; every gemma probe number is a register-sensitive "
                  "readout, and the transfer battery is gemma's primary values readout (R2-D63 refining R2-D61; the superseded "
                  "raw-render rows were a missing BOS, R2-D43 + R2-D52)")
QWEN_PROBE_NOTE = ("qwen: about half of the in-context shift is register; the remainder is a genuine within-genre values tilt "
                   "(v3: R +0.017*, C +0.028*) — R2-D63")
#: the decision-span columns' standing label
SPAN_SECONDARY_LABEL = "SECONDARY; register-sensitive unless genre-conditioned (R2-D61/R2-D63)"
#: the genre-matched probe v3 — a SEPARATE instrument, reported in its own table, never swapped in
#: as the primary. Read-only from the probe-genre lane's analysis dir; tolerant of absence.
PROBE_GENRE_DIR = STAGE / "probe_genre" / "analysis"
PROBE_V3_PROVENANCE = {"probe": "probe_v3_genre.jsonl (662 items: 266 both-refusal R / 396 both-compliance C; in-coverage 549 = R 222 / C 327)",
                       "sha256_prefix": "e2768bb4", "branch": "probe-genre-matched @ 5f00f98",
                       "superseded": "the 673-item probe_v3_genre.v0.jsonl (sha f066050c…) — max |Δ| 0.004 on full-passage cells, no CI status changed (R2-D67 update)",
                       "tables": "data/sdfbase/probe_v3_genre.scoring.{md,json} (that branch)",
                       "scores": "~/modal-stage/sdfbase/probe_genre/analysis/{shift_full,span_full}.json",
                       "fitted_cells": "probe v3g re-score of our cells (user-approved, R2-D67): <harvest>/probe_v3/<key>/<cell>.jsonl, "
                                       "stamped, per-item variant R/C — PENDING until the probe-span lane publishes them"}
#: the cells the v3g re-score covers, per base model, 1x only — plus the r2 control cells
#: (the comparison of record is full − full, so the ctl2 cells' v3g rows are read here too; ctl2_probe_v3.sh)
PROBE_V3_CELLS = ["bare", "constitution", "constv2-mix-1m__psr_ce_all", "constv2-mix-1m__lora_r4_ce",
                  "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl-100k__psr_ce_all", "constv2ctl-100k__lora_r4_ce",
                  "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
#: the genre-matched probe's OWN frozen sha (probe_v3_genre.jsonl, 662 items; probe_v3/README.md) — a v3g sidecar must
#: name it (never the v2 sha); in-coverage cardinality per variant (R 222 / C 327), enforced on every v3g pairing
PROBE_V3_SHA_FROZEN = "e2768bb4a20115497f93eaf1779579280140aed369a1fd0f57eb40dd2f0ef67a"
PROBE_V3_N_IN = {"R": 222, "C": 327}
PROBE_V3_GENRE = {"R": "refusal", "C": "compliance"}

#: the comparison of record is the FULL corpora only — constv2-mix-full
#: against the budget-matched r2 control constv2ctl2-full, same model, same arm, on every readout. 100k and every
#: smaller size, the 1M budget-mismatched rows and the round-1 control constv2ctl-100k are OUT of the analysis:
#: DEMOTED to the legacy / provenance section, never deleted — the round-1 regression (632 numbers) and the six
#: published 100k derived objects depend on those code paths staying byte-identical.
FULL_MIX_SI = "constv2-mix-full"
SCOPE_R2D111 = {
    "directive": "R2-D111: full corpora only — constv2-mix-full vs constv2ctl2-full, same model, same arm",
    "record": {"corpus_si": FULL_MIX_SI, "control_si": CTL2_SI},
    "demoted": ["constv2-mix-100k (and every smaller size)", "the 1M budget-mismatched decomposition rows",
                "the round-1 control constv2ctl-100k (register_share_100k, the 100k decomposition and its derived objects)"],
    "kept_because": "provenance: the round-1 regression and the six published 100k derived objects read these code paths",
}
#: live-cell sidecars the puller stages at <R2>/cells_all/<key>/<cell>/{train_budget_meta.json, COMPLETE}: the full pair
#: (the CE register share reads fit_record.heldout_ce_active / heldout_ce_noop and heldout_pool_sha off them, and every
#: probe row of a cell must stamp the object its sidecar names) and the 10M cells (the fresh-label evals key on the
#: live object's sha)
SIDECAR_SIS = ["constv2-mix-10m", FULL_MIX_SI, CTL2_SI]
#: fresh-label evals: MMLU + GSM8K under label r5 for the objects the continuations CHANGED — paired against
#: the r5 bare anchor within the label, keyed on identity.object_sha256, never on the label
FRESH_LABEL = "r5"
FRESH_LABEL_EXPECTED = {"qwen35_4b_base": ["constv2-mix-10m__psr_ce_all", "constv2-mix-full__psr_ce_all"]}
#: the full-scale derived objects' producer layout (scripts/modal/sdfbase_derived.py on branch derived-full @ 6708221;
#: driver $STAGE/sdfbase/derived_full_qwen.sh): /vol/out/derived/<kind>/<key>/<derived id>/… — NOTE the LoRA
#: kind dir is `lora_subtraction` (lora_subtract.py's --out-root), not `lora_subtract`; probe rows land beside the
#: fitted cells' (<key>/probe_v3/<id>.jsonl, <key>/probe_v2/<id>.jsonl; scale 1.0 only) and CE at
#: ce_score/<key>/<id>__<pool>__scale1.json on the pools the driver names (the control's held-out twin + the corpus held-out)
DERIVED_FULL_FILES = {
    "psr_paired_write": ["object.pt", "derived_meta.json", "gate_liveness.json", "gate_liveness_detail.json"],
    "psr_residual": ["object.pt", "derived_meta.json", "gate_liveness.json", "gate_liveness_detail.json", "calibration.json",
                     "calibration_controls.json", "capture_manifest.json", "report.json", "precheck_identity_report.json",
                     "doc_records.jsonl", "calibration_doc_records.jsonl"],
    "lora_subtraction": ["object.pt", "derived_meta.json"],
}
DERIVED_FULL_POOLS = ("constv2-heldout", "constv2ctl2-heldout")
#: the GLM / Nemotron doc cells are KEPT with this caveat (no refit)
WS3_TRAINING_CAVEAT = ("the GLM / Nemotron doc cells trained with an EMPTY special-token prefix (no [gMASK]<sop> / <s>); kept with "
                       "that caveat, no refit (R2-D65/R2-D67) — their chat-rendered probe and MMLU rows are unaffected")
PROBE_V3_MODELS = {"gemma4e4b_it": "gemma-it", "qwen35_4b_it": "qwen-it", "gemma4e4b_base": "gemma-base", "qwen35_4b_base": "qwen-base"}
PROBE_V3_SETS = ("old", "R", "C", "R+C")
#: decision-span statistics, SECONDARY to the pre-registered full-passage primary — optional per-row fields the
#: re-scored probe files carry (mean per-token margin, consistent − violating, over that span); absent = PENDING, never refused
SPAN_STATS = {"first16": "margin_first16", "div": "margin_div", "first_sentence": "margin_first_sentence"}
SPAN_LABEL = {"first16": "first 16 tokens", "div": "diverging span", "first_sentence": "first sentence"}
#: where the instruct-sibling in-context control rows would be read from once the probe lane stages them (the
#: gemma-it / qwen-it rows live in the probe-pairs lane's diag/out/, not in this harvest; layout to be confirmed)
INSTRUCT_PROBE_DIR = "probe_v2/{model}_instruct"

#: what each lane carries — the tables say "not in this lane" for the rest
#: instead of filing PENDING rows nobody can resolve
LANES = {
    "gemma4e4b_base": {"kind": "base", "mix_sis": MIX_SIS, "adv_sis": ADV_SIS, "ctl_sis": CTL_SIS,
                       "refits": True, "derived": True, "transfer": True, "gate_reg": True,
                       "render": "raw", "probe_legs": "descriptive (register-sensitive) on every probe statistic — full mean and "
                                                       "decision span alike (R2-D63); the transfer battery is gemma's primary values readout",
                       "positive_control": "round-1 D20/D21 chat-rendered instruct positive control: clean null (D21 caveat stands); "
                                           + GEMMA_D21_NOTE},
    "qwen35_4b_base": {"kind": "base", "mix_sis": MIX_SIS, "adv_sis": ADV_SIS, "ctl_sis": CTL_SIS,
                       "refits": True, "derived": True, "transfer": True, "gate_reg": False,
                       "render": "raw", "probe_legs": "confirmatory",
                       "positive_control": "round-1 instruct positive control passed (+0.0371 [+0.0292, +0.0446])"},
    "glm46vflash_doc": {"kind": "ws3_chat", "mix_sis": MIX_SIS_R1, "adv_sis": [], "ctl_sis": [CTL_SI],
                        "refits": False, "derived": False, "transfer": False, "gate_reg": False,
                        "render": "chat", "probe_legs": "decided from the in-context positive control (see probe_validity)",
                        "positive_control": "NONE beyond the in-context constitution row: the checkpoint IS the chat model "
                                            "(no base variant, no instruct sibling); no transfer battery exists for this lane; "
                                            "the in-context row is the SI DIGEST, which lacks the sections behind 64% of the "
                                            "sectioned items (R2-D52)"},
    "nemotron3nano4b_doc": {"kind": "ws3_chat", "mix_sis": MIX_SIS_R1, "adv_sis": [], "ctl_sis": [CTL_SI],
                            "refits": False, "derived": False, "transfer": False, "gate_reg": False,
                            "render": "chat", "probe_legs": "decided from the in-context positive control (see probe_validity)",
                            "positive_control": "NONE beyond the in-context constitution row: the checkpoint IS the chat model "
                                                "(no base variant, no instruct sibling); no transfer battery exists for this lane; "
                                                "the in-context row is the SI DIGEST, which lacks the sections behind 64% of the "
                                                "sectioned items (R2-D52)"},
}


def lane_sis(model: str) -> list[str]:
    ln = LANES[model]
    return list(ln["mix_sis"]) + list(ln["adv_sis"]) + list(ln["ctl_sis"])

# official-token budgets; the full cell's realized total is read from
# pools_summary.json when present, this is the recorded fallback
TOKENS = {"constv2-mix-1k": 1_000, "constv2-mix-10k": 10_000,
          "constv2-mix-100k": 100_000, "constv2-mix-1m": 1_000_000,
          "constv2-mix-10m": 10_000_000, "constv2-mix-full": 42_785_435,
          "constv2-adv-1k": 1_000, "constv2-adv-100k": 100_000,
          CTL_SI: 100_000, "constv2ctl-1m": 1_000_000, "constv2ctl-10m": 10_000_000,
          # the r2 control is budget-matched to mix-full BY CONSTRUCTION (build_ctl2_pools.py reads
          # the budget from mix-full's manifest) — Layout.tokens() prefers its own pool manifest's
          # official_tokens once that is staged and alarms on a mismatch
          CTL2_SI: CTL2_TOKENS}
# round-2 configs; round-1 caps kept for the superseded rows
EPOCH_CAP_R2 = {"constv2-mix-1k": 200, "constv2-mix-10k": 120,
                "constv2-mix-100k": 60, "constv2-mix-1m": 24,
                "constv2-mix-10m": 8, "constv2-mix-full": 4,
                "constv2-adv-1k": 200, "constv2-adv-100k": 60, CTL_SI: 60,
                # matched controls: caps unknown until their fits config lands
                "constv2ctl-1m": None, "constv2ctl-10m": None, CTL2_SI: CTL2_EPOCH_CAP}
# the cap-34 continuation of the three cap-24 1M cells
EPOCH_CAP_CONTINUED = 34
EPOCH_CAP_R1 = dict(EPOCH_CAP_R2, **{"constv2-mix-1m": 12})

# derived objects — cell ids used for probe/ce/mmlu rows
DERIVED_KINDS = {
    "V": ("psr_paired_write", "psr_ce_all", "{si}__psr_ce_all_minus_ctl"),
    "P_resid": ("psr_residual", "psr_ce_all",
                "{si}__psr_ce_all_minus_ctl_distilled"),
    "LoRA_sub": ("lora_subtraction", "lora_r4_ce",
                 "{si}__lora_r4_ce_minus_ctl"),
}
DERIVED_SCALES = {"constv2-mix-100k": "matched_100k (primary; PROVISIONAL round-1 control)",
                  "constv2-mix-1m": "budget_mismatched_1m (descriptive)",
                  # V / P_resid / LoRA-sub at full − full against the r2 control, derived from the
                  # CURRENT mix-full objects; ids follow the producers' `{si}__<arm>_minus_ctl*` template
                  # until the derived lane publishes them (a different id is a one-line contract change here)
                  "constv2-mix-full": "matched_full (r2 control; V / P_resid / LoRA-sub vs constv2ctl2-full — R2-D48 (e))"}
#: the control each derived scale subtracts, the control-register held-out pool P3 reads, and the
#: in-sample descriptive pool (the control's own training docs)
DERIVED_CONTROL = {"constv2-mix-100k": CTL_SI, "constv2-mix-1m": CTL_SI, "constv2-mix-full": CTL2_SI}
DERIVED_P3_POOL = {"constv2-mix-100k": "control-heldout", "constv2-mix-1m": "control-heldout",
                   "constv2-mix-full": "constv2ctl2-heldout"}
DERIVED_INSAMPLE_POOL = {"constv2-mix-100k": CTL_SI, "constv2-mix-1m": CTL_SI, "constv2-mix-full": CTL2_SI}
#: how each scale's P1–P3 are read: the pre-registered margins are confirmatory at the matched scales;
#: the 1M rows are budget-mismatched by design
DERIVED_VERDICT_SCOPE = {"constv2-mix-100k": "confirmatory", "constv2-mix-1m": "budget-mismatched",
                         "constv2-mix-full": "confirmatory"}
DERIVED_CELLS = {
    (obj, si): tmpl.format(si=si)
    for obj, (_, _, tmpl) in DERIVED_KINDS.items() for si in DERIVED_SCALES}

CE_POOLS = {
    "constv2-heldout": "corpus held-out (128 docs, the fit's own early-stop pool)",
    "control-heldout": "control-register held-out (~59k-token remainder of the "
                       "159k control corpus unused by ctl-100k; ~20 docs)",
    "constv2ctl-100k": "ctl-100k TRAINING docs (in-sample, descriptive)",
    # the r2 control lane's pools (build_ctl2_pools.py on branch ctl-full) — the frozen 128-doc
    # control held-out twin of constv2-heldout, and ctl2-full's training docs
    "constv2ctl2-heldout": "r2 control-register held-out (128 docs, the constv2-heldout twin; P3 pool for full − full)",
    "constv2ctl2-full": "ctl2-full TRAINING docs (in-sample, descriptive)",
}
# pool KEY (as used in ce_score filenames) -> file STEM of the AUTHORITATIVE pool
# manifest pairs_<stem>.manifest.json under POOLS_DIR (the control held-out pool's
# manifest is pairs_constv2ctl-heldout.manifest.json).
# The per-pool doc set / pairs sha a ce_score file is judged against comes ONLY
# from these manifests — never from a score file.
CE_POOL_MANIFEST = {
    "constv2-heldout": "constv2-heldout",
    "control-heldout": "constv2ctl-heldout",
    "constv2ctl-100k": "constv2ctl-100k",
    "constv2ctl2-heldout": "constv2ctl2-heldout",
    "constv2ctl2-full": "constv2ctl2-full",
}
#: pools whose manifests the ctl2 lane writes OUTSIDE POOLS_DIR (build_ctl2_pools.py --out pools_ctl2/);
#: everything else is under POOLS_DIR. Layout.pool_manifest_path names the exact file either way.
POOLS_CTL2_DIR = STAGE / "pools_ctl2"
CE_POOL_MANIFEST_IN_CTL2_DIR = ("constv2ctl2-heldout", "constv2ctl2-full")
CE_POOLS_CONFIRMATORY = ("constv2-heldout", "control-heldout", "constv2ctl2-heldout")

# model pins (the campaign's ONE pin registry is derive_model_config.MODEL_SPECS in
# the repo; the fits configs carry the same model/model_revision). Every probe
# sidecar's meta.model / meta.revision must equal these.
MODEL_SPECS = {
    "gemma4e4b_base": {"model": "google/gemma-4-E4B",
                       "revision": "411aa17b749aa952df1359d2dcea73917a544d9a", "render": "raw"},
    "qwen35_4b_base": {"model": "Qwen/Qwen3.5-4B-Base",
                       "revision": "1001bb4d826a52d1f399e183466143f4da7b741b", "render": "raw"},
    # doc lanes: CHAT checkpoints scored with probe render ``chat``;
    # pins from configs/sdfbase_<key>_fits.yaml on branch ws3-newmodels
    "glm46vflash_doc": {"model": "zai-org/GLM-4.6V-Flash",
                        "revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e", "render": "chat"},
    "nemotron3nano4b_doc": {"model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                            "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f", "render": "chat"},
}
#: where a lane's fits config lives when it is NOT checked out in REPO (the doc
#: configs are on branch ws3-newmodels); model_spec() reads it via ``git show``
CONFIG_BRANCH = {"glm46vflash_doc": "ws3-newmodels", "nemotron3nano4b_doc": "ws3-newmodels"}
HPO_DIR = STAGE / "hpo"

# gate_reg mini-sweep (emit_gatereg_trials)
GATEREG_MODEL = "gemma4e4b_base"
GATEREG_ARM = "psr_ce_all"
GATEREG_ROUND = 3
GATEREG_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
GATEREG_BASELINE = 0.1

# transfer battery (round-1 schema mirrored)
XFER_MODELS = {"gemma4e4b_base": "gemma4e4b", "qwen35_4b_base": "qwen35_4b"}
XFER_R2_CELLS = ["none", "constv2ctl-100k__psr_ce_all",
                 "constv2ctl-100k__lora_r4_ce"]
XFER_REGRADE_CELL = "none_r1"
XFER_ATTACKS = ["pasttense", "codeattack", "renellm", "grok420", "bon"]
#: battery namespaces, each pairing within itself only. r2: register controls +
#: the round-1 bare RESPONSES regraded in the r2 pass (`none_r1`). r3: a fresh
#: bare anchor + ALL FOUR current 1M objects (the `_r2` wave's 1M rows served the
#: round-1 objects) + the `_r2` bare RESPONSES regraded in the r3 pass
#: (`none_r2`); its drift manifests name none / none_r2 / none_r2_original,
#: the "original" grades being the r2 aggregate's. ``regrade`` is the regrade
#: cell's name (None = no drift inset in that namespace's design);
#: ``original_from`` says where the regraded cell's ORIGINAL grades live.
XFER_R3_CELLS = {m: ["none", "constv2-mix-1m__psr_ce_all", "constv2-mix-1m__lora_r4_ce"] for m in MODELS}
XFER_IT_CELLS = {
    "gemma4e4b_base": ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2-mix-1m__lora_r4_ce"],
    "qwen35_4b_base": ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"],
}
XFER_NAMESPACES = {
    "r2": {"dir": "transfer_r2", "suffix": "_r2", "regrade": "none_r1", "original_from": "r1_repo_out",
           "cells": {m: XFER_R2_CELLS for m in MODELS},
           "label": "r2 wave — same-wave bare anchor, register controls, regraded round-1 bare"},
    "r3": {"dir": "transfer_r3", "suffix": "_r3", "regrade": "none_r2", "original_from": "r2",
           "cells": XFER_R3_CELLS,
           "label": "r3 wave — fresh bare anchor + all four current 1M objects (R2-D32) + the r2 bare regraded (R2-D33)"},
    # the instruct-direct wave. Objects trained ON the instruct checkpoint (keys <model>_it_doc, cells named by the
    # constitutional pseudo-instruction constv2-mix-full) are applied to that same model directly — no sibling staging. gemma also
    # carries the NEW 1M LoRA object (chained continuation, sha 370c0755…) whose r3 row is one generation stale. Regrade cell = the r3
    # bare responses regraded in the it judge pass. PENDING by path until the wave lands.
    "it": {"dir": "transfer_it", "suffix": "_it", "regrade": "none_r3", "original_from": "r3",
           "cells": XFER_IT_CELLS,
           "label": "it wave — fresh bare anchor + the four instruct-direct full-corpus objects + gemma's re-run 1M LoRA + the r3 bare regraded"},
}
#: the instruct-direct objects were trained on the INSTRUCT checkpoints under their own model keys. The battery
#: is per served model, so their rows sit under the BASE lane's transfer slug with cell ids named by the constitutional
#: pseudo-instruction (constv2-mix-full__<arm>) — the SAME ids the base lanes use for their mix-full cells. Object
#: identity therefore binds per (namespace, cell): in `it` the constv2-mix-full cells belong to <base key>_it_doc, every
#: other battery cell to the served base key itself. The it_doc keys' identity-bearing files are read from
#: <R2>/probe_v2/<it_doc key>/<cell>.jsonl.meta.json (stamps) and <R2>/mmlu/<it_doc key>/<label>/ — the lane layout.
XFER_IT_DOC_KEY = {"gemma4e4b_base": "gemma4e4b_it_doc", "qwen35_4b_base": "qwen35_4b_it_doc"}
XFER_IT_DOC_CELL_PREFIX = "constv2-mix-full__"


def xfer_object_owner(ns: str, model: str, cell: str) -> str:
    """The model key whose fitted object a battery cell served — the key its identity-bearing
    files live under (probe stamps, MMLU identities). Only the `it` namespace's constv2-mix-full
    cells belong to a key other than the served base model."""
    if ns == "it" and cell.startswith(XFER_IT_DOC_CELL_PREFIX):
        return XFER_IT_DOC_KEY[model]
    return model


#: the continuation driver's fit-call manifests (<STAGE>/fit_calls_*.json): {cells: [{model, cell_id, expect_fingerprint,
#: continue_cap, continue_lr_mult, call_id, spawn_time, status, continued_from: {prior_fingerprint, ...}}]}. A row whose
#: continued_from.prior_fingerprint is a staged snapshot's fit_fingerprint says that snapshot is the publication a re-issue
#: STILL RUNNING resumed from (the live cell re-publishes only when it finishes) — "continuation in flight", not foreign.
FIT_CALLS_GLOB = "fit_calls_*.json"

REFITS_SCHEMA = {"<model_key>": ["<cell id refit this round>", "..."],
                 "_doc": "operator-declared refit set; see module docstring"}

DRIFT_MANIFEST_SCHEMA = {
    "path": "<transfer_r2 dir>/drift_manifests/<attack>.json  (one per attack incl. bon)",
    "attack": "<attack>",
    "judge": {"model": "claude-haiku-4-5", "rubric": "strongreject_v1",
              "pass_id": "<id of the fresh r2 grading pass, e.g. sha256 of the judge cache at close>"},
    "cells": {
        "none": {"pass_id": "<== judge.pass_id>",
                 "rows": [{"goal_id": "<goal>", "response_sha256":
                           "sha256(utf-8 response text); BoN: sha256 of the N candidate "
                           "responses joined by '\\n\\x00' in draw order"}]},
        "none_r1": {"pass_id": "<== judge.pass_id>",
                    "source": "round-1 bare RESPONSES copied read-only into the r2 namespace",
                    "rows": ["... same shape; ordered by goal_id ..."]},
        "none_r1_original": {"pass_id": "<the round-1 grading pass id>",
                             "judge": {"model": "...", "rubric": "..."},
                             "rows": ["... MUST be identical ordered (goal_id, response_sha256) "
                                      "to cells.none_r1 — byte-identical responses ..."]},
    },
    "_checks": ["judge-drift leg: rows(none_r1) == rows(none_r1_original) exactly, else REFUSED",
                "generation-drift leg: cells.none.pass_id == cells.none_r1.pass_id == judge.pass_id, "
                "else REFUSED",
                "goal sets in every cell's rows == the aggregate's per_goal keys, else REFUSED"],
}

CALIBRATION_SCHEMA = {
    "strata": {"<source>/<position_class>": {
        "ed": "float — 1 - MSE(student, teacher)/MSE(bare, teacher)",
        "ed_ci95": "[lo, hi] doc-level bootstrap (1,000 resamples)",
        "ed_attain": "float — min over the two planted recoverable controls",
        "ed_attain_components": {"R_mix_empty": "float", "R_mixC_mix": "float"},
        "teacher_bare_mse": "float — MSE(bare, teacher) on the holdout",
        "floor_mse": "float — repeated bare/bare capture MSE",
    }},
    "gate": {"verdict": "recovered | not_recovered | unresolvable",
             "sentence": "the exact failure sentence, if any"},
    "t_arith_rms": "float — arithmetic-residual inconsistency (diagnostic)",
    "n_holdout_docs": {"<stratum>": "int"},
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Optional[Path]):
    if p is None or not Path(p).exists():
        return None
    return json.loads(Path(p).read_text())


def read_jsonl(p: Optional[Path]):
    if p is None or not Path(p).exists():
        return None
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


@dataclass
class Layout:
    """Resolves every input the pipeline reads. ``None`` = not staged."""
    harvest: Path = R2_HARVEST_DEFAULT
    harvest_r1: Path = R1_HARVEST
    r1_derived: Path = R1_DERIVED
    r1_transfer: Path = R1_TRANSFER
    pools_summary: Path = R1_POOLS
    pools_dir: Path = POOLS_DIR
    pools_ctl2_dir: Path = POOLS_CTL2_DIR
    probe_genre_dir: Path = PROBE_GENRE_DIR
    repo: Path = REPO
    xfer_private: Path = XFER_PRIVATE
    hpo_dir: Path = HPO_DIR
    fit_calls_dir: Path = STAGE          # the continuation driver's fit_calls_*.json manifests (outside the harvest root)
    resolved: dict = field(default_factory=dict)

    def lane(self, model: str) -> dict:
        if model not in LANES:
            raise SystemExit(f"[r2] no lane definition for {model!r} — add it to r2_paths.LANES")
        return LANES[model]

    def _note(self, key: str, p: Optional[Path], looked: Optional[Path] = None
              ) -> Optional[Path]:
        """Record what was looked for (so a PENDING row can name the path the
        operator must stage) and return the path only if it exists."""
        self.resolved[key] = str(p if p is not None else looked)
        return p

    def _find(self, key: str, p: Path) -> Optional[Path]:
        return self._note(key, p if p.exists() else None, looked=p)

    # ---- probe -----------------------------------------------------------
    def probe(self, model: str, cell: str, scale: float = 1.0,
              root: Optional[Path] = None) -> Optional[Path]:
        root = root or self.harvest
        d = "probe_v2" if float(scale) == 1.0 else f"probe_v2_x{float(scale):g}"
        p = root / d / model / f"{cell}.jsonl"
        # the round-1 lookup keeps its own key: a PENDING row names the r2 path to stage, never round 1's
        tag = "probe" if root == self.harvest else "probe_r1"
        return self._find(f"{tag}/{model}/{cell}@x{scale:g}", p)

    def probe_r1(self, model: str, cell: str) -> Optional[Path]:
        return self.probe(model, cell, root=self.harvest_r1)

    def probe_meta(self, p: Path) -> Optional[dict]:
        return read_json(Path(str(p) + ".meta.json"))

    # ---- fit reports -----------------------------------------------------
    def report(self, model: str, cell: str,
               root: Optional[Path] = None) -> Optional[Path]:
        root = root or self.harvest
        p = root / "reports" / model / f"report_{cell}_all.json"
        tag = "report" if root == self.harvest else "report_r1"
        return self._find(f"{tag}/{model}/{cell}", p)

    def report_r1(self, model: str, cell: str) -> Optional[Path]:
        return self.report(model, cell, root=self.harvest_r1)

    def summary(self, root: Optional[Path] = None) -> Optional[Path]:
        p = (root or self.harvest) / "summary.json"
        return p if p.exists() else None

    # ---- mmlu ------------------------------------------------------------
    def mmlu_dir(self, model: str, label: Optional[str]) -> Optional[Path]:
        base = self.harvest / "mmlu" / model
        if label is None:
            return base if base.exists() else None
        d = base / label
        return d if d.exists() else None

    def mmlu_aggregate(self, model: str, cell: str,
                       label: Optional[str]) -> Optional[Path]:
        cands = []
        if label is not None:
            # base_mmlu's naming first (label in dir AND filename), then the bare twin
            cands.append(self.harvest / "mmlu" / model / label / f"mmlu_{label}_{cell}.json")
            cands.append(self.harvest / "mmlu" / model / label / f"mmlu_{cell}.json")
            cands.append(self.harvest / "mmlu" / model / f"mmlu_{cell}.{label}.json")
        else:
            cands.append(self.harvest / "mmlu" / model / f"mmlu_{cell}.json")
        for p in cands:
            if p.exists():
                return self._note(f"mmlu/{model}/{label}/{cell}", p)
        return self._note(f"mmlu/{model}/{label}/{cell}", None, looked=cands[0])

    def mmlu_per_question(self, model: str, cell: str,
                          label: Optional[str]) -> Optional[Path]:
        cands = []
        if label is not None:
            cands.append(self.harvest / "mmlu" / model / label
                         / f"mmlu_{label}_{cell}.per_question.jsonl")
            cands.append(self.harvest / "mmlu" / model / label
                         / f"mmlu_{cell}.per_question.jsonl")
            cands.append(self.harvest / "mmlu" / model
                         / f"mmlu_{cell}.{label}.per_question.jsonl")
        else:
            cands.append(self.harvest / "mmlu" / model
                         / f"mmlu_{cell}.per_question.jsonl")
        for p in cands:
            if p.exists():
                return self._note(f"mmlu_pq/{model}/{label}/{cell}", p)
        return self._note(f"mmlu_pq/{model}/{label}/{cell}", None, looked=cands[0])

    def mmlu_labels(self, model: str) -> list[str]:
        """Run labels with at least one aggregate staged (an empty label dir —
        the pull script creates r1/r3 for every lane — is not a label)."""
        base = self.harvest / "mmlu" / model
        if not base.exists():
            return []
        labels = sorted(d.name for d in base.iterdir() if d.is_dir()
                        and any(not p.name.endswith("summary.json") for p in d.glob("mmlu_*.json")))
        flat = sorted({p.name.split(".")[-2] for p in base.glob("mmlu_*.*.json")
                       if p.name.count(".") >= 2
                       and not p.name.endswith("per_question.jsonl")})
        return sorted(set(labels) | set(flat))

    # ---- live-cell sidecars + the genre-matched probe (full − full; fresh-label identity) ----
    #   <R2>/cells_all/<model>/<cell>/{train_budget_meta.json, COMPLETE} (the live bank's own sidecar, staged by
    #   harvest_pull_r2.sh for SIDECAR_SIS × ARMS); <R2>/probe_v3/<model>/<cell>.jsonl (+ .meta.json)
    def cell_sidecar(self, model: str, cell: str) -> Optional[Path]:
        return self._find(f"sidecar/{model}/{cell}", self.harvest / "cells_all" / model / cell / "train_budget_meta.json")

    def cell_complete(self, model: str, cell: str) -> Optional[Path]:
        return self._find(f"complete/{model}/{cell}", self.harvest / "cells_all" / model / cell / "COMPLETE")

    def probe_v3(self, model: str, cell: str) -> Optional[Path]:
        return self._find(f"probe_v3/{model}/{cell}", self.harvest / "probe_v3" / model / f"{cell}.jsonl")

    # ---- ce_score --------------------------------------------------------
    def ce_score(self, model: str, cell: str, pool: str,
                 scale: float = 1.0) -> Optional[Path]:
        p = (self.harvest / "ce_score" / model
             / f"{cell}__{pool}__scale{float(scale):g}.json")
        return self._find(f"ce_score/{model}/{cell}/{pool}@x{scale:g}", p)

    # ---- derived ---------------------------------------------------------
    def derived_dir(self, obj: str, model: str, si: str) -> Optional[Path]:
        kind, _arm, tmpl = DERIVED_KINDS[obj]
        d = self.harvest / "derived" / kind / model / tmpl.format(si=si)
        return self._find(f"derived/{obj}/{model}/{si}", d)

    # ---- gsm8k — the <R2>/gsm8k/README.md layout contract (2026-09-03) -----
    #   <R2>/gsm8k/<model>/<label>/gsm8k_<label>_<cell>.json, labels r2 (1x) / r2_x2 (2x)
    def gsm8k_dir(self, model: str, label: Optional[str]) -> Optional[Path]:
        base = self.harvest / "gsm8k" / model
        if label is None:
            return base if base.exists() else None
        d = base / label
        return d if d.exists() else None

    def gsm8k_labels(self, model: str) -> list[str]:
        base = self.harvest / "gsm8k" / model
        return sorted(d.name for d in base.iterdir() if d.is_dir()) if base.exists() else []

    def gsm8k_aggregate(self, model: str, cell: str, label: str) -> Optional[Path]:
        p = self.harvest / "gsm8k" / model / label / f"gsm8k_{label}_{cell}.json"
        return self._find(f"gsm8k/{model}/{label}/{cell}", p)

    def gsm8k_per_question(self, model: str, cell: str, label: str) -> Optional[Path]:
        p = self.harvest / "gsm8k" / model / label / f"gsm8k_{label}_{cell}.per_question.jsonl"
        return self._find(f"gsm8k_pq/{model}/{label}/{cell}", p)

    def gsm8k_readme(self) -> Optional[Path]:
        p = self.harvest / "gsm8k" / "README.md"
        return p if p.exists() else None

    def derived_file(self, obj: str, model: str, si: str,
                     name: str) -> Optional[Path]:
        d = self.derived_dir(obj, model, si)
        if d is None:
            return None
        p = d / name
        return p if p.exists() else None

    # ---- gate_reg --------------------------------------------------------
    def gatereg_design(self) -> Optional[Path]:
        p = (self.harvest / "hpo/trials"
             / f"r{GATEREG_ROUND}_{GATEREG_MODEL}_{GATEREG_ARM}_all.json")
        return self._find("gatereg/design", p)

    def gatereg_trial_summary(self, k: int) -> Optional[Path]:
        p = (self.harvest / f"hpo/results/r{GATEREG_ROUND}/{GATEREG_MODEL}"
             / f"{GATEREG_ARM}_all" / f"r{GATEREG_ROUND}_t{k:02d}/summary.json")
        return self._find(f"gatereg/trial/{k}", p)

    def gatereg_trial_liveness(self, k: int) -> Optional[Path]:
        p = (self.harvest / f"hpo/results/r{GATEREG_ROUND}/{GATEREG_MODEL}"
             / f"{GATEREG_ARM}_all" / f"r{GATEREG_ROUND}_t{k:02d}"
             / "gate_liveness.json")
        return p if p.exists() else None

    # ---- snapshots: pre-continuation publications, one bank per continuation cap
    #      (_r2cap24, _r2cap8, _r2cap4). The bank a live cell resumed from is named by its marker
    #      (continued_from.snapshot) and resolved through parse_snapshot_marker — never assumed.
    def snapshot_banks(self, model: str) -> list[str]:
        """Bank suffixes staged for this model (sorted by cap), e.g. ['_r2cap4', '_r2cap8', '_r2cap24']."""
        root = self.harvest / "snapshots"
        if not root.exists():
            return []
        out = []
        for d in root.iterdir():
            m = re.match(rf"^{re.escape(model)}(_r2cap(\d+)([a-z]?))$", d.name)
            if d.is_dir() and m and (d / "cells_all").exists():
                out.append((int(m[2]), m[3], m[1]))
        return [suf for _cap, _var, suf in sorted(out)]

    def snapshot_bank(self, model: str, suffix: str = SNAPSHOT_SUFFIX) -> Optional[Path]:
        d = self.harvest / "snapshots" / f"{model}{suffix}"
        return self._find(f"snapshot_bank/{model}{suffix}", d)

    def snapshot_cell(self, model: str, cell: str, suffix: str = SNAPSHOT_SUFFIX) -> Optional[Path]:
        """The snapshotted cell dir (COMPLETE, train_budget_meta.json,
        snapshot_of.json, …) or None."""
        d = self.harvest / "snapshots" / f"{model}{suffix}" / "cells_all" / cell
        return self._find(f"snapshot/{model}{suffix}/{cell}", d)

    def snapshot_cell_expected(self, model: str, cell: str, suffix: str) -> Path:
        """Where the marker's snapshot must be staged (exists or not) — the path a PENDING row names."""
        return self.harvest / "snapshots" / f"{model}{suffix}" / "cells_all" / cell

    def snapshot_cells(self, model: str, suffix: str = SNAPSHOT_SUFFIX) -> list[str]:
        d = self.harvest / "snapshots" / f"{model}{suffix}" / "cells_all"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "COMPLETE").exists())

    def snapshot_cells_all_banks(self, model: str) -> list[tuple[str, str]]:
        """Every (suffix, cell) snapshotted for this model, across all staged banks."""
        return [(suf, c) for suf in self.snapshot_banks(model) for c in self.snapshot_cells(model, suf)]

    def snapshot_report(self, model: str, cell: str, suffix: str = SNAPSHOT_SUFFIX) -> Optional[Path]:
        p = (self.harvest / "snapshots" / f"{model}{suffix}" / "reports"
             / f"report_{cell}_all.json")
        return self._find(f"snapshot_report/{model}{suffix}/{cell}", p)

    # ---- transfer --------------------------------------------------------
    def transfer_ns_dir(self, model: str, ns: str = "r2") -> Optional[Path]:
        """<harvest>/<XFER_NAMESPACES[ns].dir>/<slug> when it holds at least
        one battery aggregate, else None. Deliberately NO fallback to the
        battery's repo out dir: every battery row is read through
        the harvest root the caller passed in, so a harvest without the
        namespace stays PENDING instead of silently resolving against live
        files elsewhere."""
        slug = XFER_MODELS[model]
        sub = XFER_NAMESPACES[ns]["dir"]
        d = self.harvest / sub / slug
        if d.exists() and any(d.glob("two_si_*_aggregate.json")):
            return self._note(f"{sub}/{slug}", d)
        return self._note(f"{sub}/{slug}", None, looked=d)

    def transfer_ns_aggregate(self, model: str, attack: str, ns: str = "r2") -> Optional[Path]:
        d = self.transfer_ns_dir(model, ns)
        if d is None:
            return None
        p = d / f"two_si_{attack}_aggregate.json"
        return p if p.exists() else None

    def transfer_ns_private(self, model: str, ns: str = "r2") -> Path:
        return self.xfer_private / f"sdfbase_transfer_{XFER_MODELS[model]}{XFER_NAMESPACES[ns]['suffix']}"

    def transfer_controls_summary(self, model: str, cell: str, ns: str = "r2") -> Optional[Path]:
        """<ns dir>/controls/<cell>/controls_summary.json — the capability leg's
        own provenance (artifact_sha256 = the object it served)."""
        slug = XFER_MODELS[model]
        p = self.harvest / XFER_NAMESPACES[ns]["dir"] / slug / "controls" / cell / "controls_summary.json"
        return self._find(f"{XFER_NAMESPACES[ns]['dir']}/{slug}/controls_summary/{cell}", p)

    # round-2-namespace wrappers (the names the rest of the pipeline grew up with)
    def transfer_r2_dir(self, model: str) -> Optional[Path]:
        return self.transfer_ns_dir(model, "r2")

    def transfer_r2_aggregate(self, model: str, attack: str) -> Optional[Path]:
        return self.transfer_ns_aggregate(model, attack, "r2")

    def transfer_r2_private(self, model: str) -> Path:
        return self.transfer_ns_private(model, "r2")

    def transfer_r1_aggregate(self, model: str, attack: str) -> Optional[Path]:
        slug = XFER_MODELS[model]
        p = self.repo / "out" / f"sdfbase_transfer_{slug}" / f"two_si_{attack}_aggregate.json"
        return self._find(f"transfer_r1/{slug}/{attack}", p)

    def drift_manifest(self, model: str, attack: str, ns: str = "r2") -> Optional[Path]:
        slug = XFER_MODELS[model]
        p = self.harvest / XFER_NAMESPACES[ns]["dir"] / slug / "drift_manifests" / f"{attack}.json"
        return self._find(f"drift_manifest/{ns}/{slug}/{attack}", p)

    # ---- doc-lane HPO provenance (outside the harvest root; Layout.hpo_dir) --
    def hpo_ws3_winners(self) -> Optional[Path]:
        return self._find("hpo/ws3_winners_fold", self.hpo_dir / "ws3_winners_fold.json")

    def hpo_fit_json(self, model: str, arm: str) -> Optional[Path]:
        """The pooled-rounds fit summary when one exists (r1+2_fit.json), else
        the round-1 one."""
        d = self.hpo_dir / "results" / "r1" / model / f"{arm}_all"
        for name in ("r1+2_fit.json", "r1_fit.json"):
            p = d / name
            if p.exists():
                return self._note(f"hpo/fit/{model}/{arm}", p)
        return self._note(f"hpo/fit/{model}/{arm}", None, looked=d / "r1_fit.json")

    # ---- pools / refits / pins ---------------------------------------------
    def pool_manifest_path(self, pool: str) -> Path:
        """The AUTHORITATIVE manifest for a ce_score pool key, via the explicit
        CE_POOL_MANIFEST mapping (control-heldout -> pairs_constv2ctl-heldout).
        Unknown keys map to a path that cannot exist."""
        stem = CE_POOL_MANIFEST.get(pool)
        base = self.pools_ctl2_dir if pool in CE_POOL_MANIFEST_IN_CTL2_DIR else self.pools_dir
        p = base / (f"pairs_{stem}.manifest.json" if stem else f"UNREGISTERED_POOL_{pool}.manifest.json")
        self._find(f"pool_manifest/{pool}", p)
        return p

    def pool_manifest(self, pool: str) -> Optional[dict]:
        p = self.pool_manifest_path(pool)
        return read_json(p) if p.exists() else None

    def model_spec(self, model: str) -> dict:
        """{model, revision} for a model key: the MODEL_SPECS pin, cross-checked
        against the repo's fits config (configs/sdfbase_<key>_fits.yaml) when
        that file is present — a drift between the two is fatal, never
        silently resolved."""
        spec = MODEL_SPECS.get(model)
        if spec is None:
            raise SystemExit(f"[r2] no model pin for {model!r} — add it to r2_paths.MODEL_SPECS")
        cfg = self.repo / "configs" / f"sdfbase_{model}_fits.yaml"
        text, source = None, None
        if cfg.exists():
            text, source = cfg.read_text(), str(cfg)
        elif model in CONFIG_BRANCH and (self.repo / ".git").exists():
            # the lane's config is committed on another branch (ws3-newmodels);
            # read it from git rather than skipping the cross-check
            import subprocess
            ref = f"{CONFIG_BRANCH[model]}:configs/sdfbase_{model}_fits.yaml"
            try:
                text = subprocess.run(["git", "-C", str(self.repo), "show", ref], check=True,
                                      capture_output=True, text=True).stdout
                source = f"git show {ref}"
            except (subprocess.CalledProcessError, OSError):
                text = None
        self.resolved[f"fits_config/{model}"] = source or "not cross-checked (config not found)"
        if text is not None:
            got = {}
            for line in text.splitlines():
                for key, field in (("model:", "model"), ("model_revision:", "revision")):
                    if line.startswith(key):
                        got[field] = line.split(":", 1)[1].strip().strip("'\"")
            for field in ("model", "revision"):
                if field in got and got[field] != spec[field]:
                    raise SystemExit(f"[r2] {source} {field}={got[field]!r} != MODEL_SPECS pin "
                                     f"{spec[field]!r} for {model} — the pins drifted; refusing")
        return spec

    def refits(self) -> dict:
        """{model: [cell, ...]} declared refits; {} when refits.json is absent."""
        p = self.harvest / "refits.json"
        self._find("refits", p)
        j = read_json(p) or {}
        return {k: list(v) for k, v in j.items() if not k.startswith("_")}

    # ---- misc ------------------------------------------------------------
    def continuation_events(self) -> Optional[Path]:
        p = self.harvest / "continuation_events.json"
        return self._find("continuation_events", p)

    def fit_call_manifests(self) -> list[Path]:
        """Every continuation-driver fit-call manifest (<fit_calls_dir>/fit_calls_*.json); [] when none."""
        return sorted(self.fit_calls_dir.glob(FIT_CALLS_GLOB)) if self.fit_calls_dir.exists() else []

    def fit_call_rows(self):
        """(manifest path, cell row) for every `cells` row of every fit-call manifest."""
        for p in self.fit_call_manifests():
            for row in ((read_json(p) or {}).get("cells") or []):
                if isinstance(row, dict):
                    yield p, row

    def tokens(self) -> dict:
        """Official-token budgets, the full cell's realized total from the
        pool summary when it exists."""
        t = dict(TOKENS)
        ps = read_json(self.pools_summary)
        if ps:
            for si, c in ps.get("cells", {}).items():
                if si in t and c.get("official_tokens"):
                    t[si] = int(c["official_tokens"])
        # the r2 control's own pool manifest (pools_ctl2/pairs_constv2ctl2-full.manifest.json) once staged
        man = read_json(self.pool_manifest_path(CTL2_SI))
        if man and man.get("official_tokens"):
            t[CTL2_SI] = int(man["official_tokens"])
        return t

    def ctl2_tokens_source(self) -> str:
        p = self.pool_manifest_path(CTL2_SI)
        return (f"{p} (official_tokens)" if p.exists()
                else f"recorded R2-D48 budget match ({CTL2_TOKENS:,} = mix-full); manifest not staged at {p}")


def print_schema() -> None:
    print(__doc__)
    print("PROBE_SHA_FROZEN =", PROBE_SHA_FROZEN, " PROBE_N =", PROBE_N)
    print("MODEL_SPECS =", json.dumps(MODEL_SPECS, indent=1))
    print("CE_POOL_MANIFEST (pool key -> pairs_<stem>.manifest.json under", str(POOLS_DIR) + ") =",
          json.dumps(CE_POOL_MANIFEST, indent=1), " confirmatory:", list(CE_POOLS_CONFIRMATORY))
    print("REFITS_SCHEMA (<R2>/refits.json) =", json.dumps(REFITS_SCHEMA, indent=1))
    print("DRIFT_MANIFEST_SCHEMA =", json.dumps(DRIFT_MANIFEST_SCHEMA, indent=1))
    print("CALIBRATION_SCHEMA =", json.dumps(CALIBRATION_SCHEMA, indent=1))
    print("DERIVED_CELLS =", json.dumps({f"{k[0]}@{k[1]}": v
                                          for k, v in DERIVED_CELLS.items()},
                                         indent=1))
    print("LANES (what each model lane carries; everything else reads 'not in this lane') =",
          json.dumps(LANES, indent=1))
    print("CTL_SIS (register controls; matched ids PENDING until staged) =", CTL_SIS,
          " MATCHED_CTL =", json.dumps(MATCHED_CTL), " DERIVED_CONTROL =", json.dumps(DERIVED_CONTROL),
          " DERIVED_P3_POOL =", json.dumps(DERIVED_P3_POOL))
    print("SNAPSHOT banks: <harvest>/snapshots/<model_key>_r2cap<N>/cells_all/<cell>, N = the prior cap named by the "
          "live report's continued_from.snapshot (R2-D20: 24; R2-D44: 8 for 10M, 4 for mix-full)")
    print("XFER_NAMESPACES =", json.dumps(XFER_NAMESPACES, indent=1))
    print("SNAPSHOT layout: <R2>/snapshots/<model>" + SNAPSHOT_SUFFIX
          + "/{cells_all/<cell>/COMPLETE, reports/report_<cell>_all.json}")
    print("HPO_DIR (WS3 winners + fit summaries) =", str(HPO_DIR))


if __name__ == "__main__":
    print_schema()
