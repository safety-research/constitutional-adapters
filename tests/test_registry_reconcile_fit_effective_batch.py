"""`registry reconcile fit-effective-batch` (reconcile.py): an object row whose `fit_effective_batch` is the micro-batch a sidecar hook read
(`resolved_batch_size`) where the row's own `fit_hyperparameters_json` says the optimizer width (`batch_size x accum_steps = effective_batch`)
is restated from its own recorded block, as one same-id `object` record written by the registry owner into
`records/registry_owner/object__<sha16>/`: the required identity fields and the kind's pairing columns copied verbatim, the recorded value,
the row's whole `field_confidence_json` with the column's entry, a marker under `extra`; `status` and every other column omitted (per-column
latest statement: they keep their folded value and stamp). The ambiguous class (the column NULL beside a recorded value: a dose copy the
fit-sidecars adapter has not yet restated as its parent's batch, or left NULL by name -- the rule of 2026-10-13 makes the bank's
re-emission the correction, from the one record copied beside the copy, told apart by the row's own confidence entry) is listed and never
corrected, and a dose copy whose derived value equals its recorded block is consistent; a disagreeing row whose column is marked `derived`
is not the hook's statement and is unclassified, refused by name; every derived dose copy is read against its PARENT object row (the
registry-level cross-check, read-only: a disagreement is listed by name, `--verify` exit 3, `--apply` writes nothing for it; a parent row
absent from the registry is not a disagreement and is said so); a disagreement outside the hook's signature, or one whose batch values are not all
integers in the record's own type (a recorded "auto", a 64.5), is unclassified and refused by name with the value and its type -- nothing
coerced, nothing truncated, nothing dropped from the report. Folded and rebuilt through the real harvester on a temp root; the CLI's exit
codes pinned; the hold behind an unfolded object record.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import reconcile as RC
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL


def _sha(tag: str) -> str:
    return hashlib.sha256(f"synthetic object {tag}".encode()).hexdigest()


#: the hook's self-contradicting block: the micro-batch 1, the accumulation 64, the trainer's effective_batch 64
HOOK_BLOCK = {"accum_steps": 64, "batch_size": 1, "resolved_batch_size": 1, "effective_batch": 64, "lr": 0.0064, "max_len": 8192, "positions": "all"}
#: a confidence entry the live row carries on ANOTHER column: the record carries it forward whole (the object table has no per-column merge)
OTHER_ENTRY = {"fit_thinking_render": {"confidence": "recorded", "source": "the trainer's sidecar"}}
(SHA_EPOCH, SHA_DOSE, SHA_AMBIG, SHA_FINE, SHA_ODD, SHA_NONE, SHA_AUTO, SHA_HALF, SHA_RULED, SHA_BYNAME, SHA_DERIVED_ODD, SHA_STALE, SHA_BYNAME2, SHA_AMBIG2,
 SHA_AGREE, SHA_VIACOL, SHA_PCHECK, SHA_ORPHAN, SHA_NOWHERE, SHA_NOKEY, SHA_NULLKEY, SHA_NOBLOCK, SHA_UNSAID, SHA_ANCHORISH,
 SHA_PFLOAT, SHA_PBOOL, SHA_CFLOAT, SHA_CBOOL, SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE) = (
    _sha(t) for t in ("epoch", "dose", "ambiguous", "fine", "odd", "nothing", "auto", "half", "ruled", "byname", "derived-odd", "stale", "byname-2", "ambiguous-2",
                      "agree", "via-column", "parent-check", "orphan", "nowhere", "no-key", "null-key", "no-block", "unsaid", "anchorish",
                      "parent-float", "parent-bool", "copy-float", "copy-bool", "statement-float", "statement-no-key", "statement-null", "statement-stale"))


def derived_entry(parent):
    """The fit-sidecars adapter's entry for a dose copy's batch under the ruling of 2026-10-13, word for word (`fit_sidecars.dose_batch_derived_source`;
    the sidecar hook's dose path writes the same words): derived from the parent's record copied beside the copy."""
    return {"confidence": "derived", "source": f"the parent's effective batch ({parent[:12]}…), read from train_budget_meta.json:effective_batch (the PARENT's fit "
                                               "record copied beside the copy); a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13)"}


DERIVED_ENTRY = derived_entry(synth.OBJECT_SHA)


def byname_reason(parent):
    """The adapter's reason for a dose copy it left NULL by name: the copied record names an integral float (the block carries the 64.0 verbatim)."""
    return (f"a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the parent's fit record copied beside the copy "
            f"(train_budget_meta.json, parent {parent[:12]}…) names effective_batch 64.0 (float), not an integer; nothing is coerced; left NULL")


BYNAME_REASON = byname_reason(synth.OBJECT_SHA)
BYNAME_ENTRY = {"confidence": "unverified", "source": BYNAME_REASON, "reason": BYNAME_REASON}
#: the adapter's entry for a SECOND copy of the same bucket it left NULL by name -- a copy of ANOTHER parent (FINE), so its reason differs in the
#: parent it names: the two share (writer, recorded, model, unverified) and only their ids and reasons tell them apart in the report
BYNAME_REASON_2 = byname_reason(SHA_FINE)
BYNAME_ENTRY_2 = {"confidence": "unverified", "source": BYNAME_REASON_2, "reason": BYNAME_REASON_2}
#: the block of a copy the adapter left NULL by name: the copied record's 64.0, carried verbatim
BYNAME_BLOCK = {**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 64.0}


def nokey_reason(parent):
    """The adapter's reason for a copy whose copied record names no `effective_batch` (the key absent or null): the block then has none either."""
    return (f"a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the parent's fit record copied beside the copy "
            f"(train_budget_meta.json, parent {parent[:12]}…) names no effective_batch; a record present and naming none is read as naming none; left NULL")


def noblock_reason(parent):
    """The adapter's reason for a copy carrying no `train_budget_meta.json` (a derived parent's dose child): the row then has no block at all."""
    return (f"a dose-scaled copy was not fit, so its effective batch is its PARENT's (the user 2026-10-13), and the copy carries no train_budget_meta.json "
            f"(the parent {parent[:12]}… is a derived object with no fit record of its own to be copied beside the copy); left NULL")


def unverified_entry(why):
    return {"confidence": "unverified", "source": why, "reason": why}


#: a row of ANOTHER derivation left NULL by its own writer (the subtraction anchor whose parents' records could not be reached): the same
#: `unverified` shape beside a NULL column and no block -- its own writer's reading, outside this pass's class
ANCHORISH_REASON = (f"a subtraction has no batch of its own; parent_pos ({SHA_EPOCH[:12]}…), parent_neg ({SHA_FINE[:12]}…) are not staged in a bank "
                    "on this machine; left NULL")
#: why ODD (stated 2, the block's micro-batch 1) is unclassified: the signature's first failing condition, as the pass words it
ODD_REASON = "fit_effective_batch 2 is not the block's resolved_batch_size 1: the sidecar hook's signature does not hold"
#: why a recorded value that is not an integer in its own type is unclassified: the value and its type, nothing coerced
AUTO_REASON = "fit_hyperparameters_json.effective_batch is 'auto' (str), not an integer; nothing is coerced"
HALF_REASON = "fit_hyperparameters_json.effective_batch is 64.5 (float), not an integer; nothing is coerced"
UNCLASSIFIED_HEAD = ("unclassified (the two values disagree and the hook's signature does not hold, a batch value is not an integer, or the column is the "
                     "adapter's derived or NULL-by-name statement, not the hook's; refused by name, never corrected)")
#: why a value standing beside the adapter's `unverified` (NULL-by-name) entry is unclassified: a retained earlier statement, never the hook's
STALE_REASON_HEAD = "fit_effective_batch is the adapter's NULL-by-name statement with a value still standing (field_confidence_json: unverified -- "


def _epoch_checkpoint(sha, role="epoch_1", **over):
    """A hook-written epoch checkpoint (the live class's shape): a fitted ReLU-steer object, status reference, the hook's block."""
    o = synth.object_lora(sha, "constitution_relu", "constitution")
    o.update(object_kind="relu_steer", status="reference", named_role=role, fit_effective_batch=1, fit_hyperparameters_json=dict(HOOK_BLOCK),
             field_confidence_json=dict(OTHER_ENTRY), fit_rank=None, fit_alpha=None)
    o.update(over)
    return {k: v for k, v in o.items() if v is not None}


def _dose_copy(sha, parent, writer="scripts/sdfbase/scale_relu_dose.py", **over):
    """A dose-scaled copy (`derived` / `dose_scale`): the parent's block beside it, the fit columns the kind requires NULL left out."""
    o = {"object_id": sha, "object_kind": "relu_steer", "paper_label": "control_relu", "campaign": synth.CAMPAIGN, "object_role": "control",
         "fit_target_kind": "derived", "fit_thinking_render": "off", "fit_objective": "not_applicable", "fit_positions": "all",
         "fit_seed_confidence": "not_applicable", "derivation": "dose_scale", "parent_object_ids_json": [parent], "base_scale": 0.75,
         "fit_model_id": synth.FIT_MODEL_ID, "fit_layers_json": [4, 5, 6], "fit_layer_index_convention": "zero_based_decoder_block", "fit_layer_count": 3,
         "fit_writer": writer, "artifact_sha256": sha, "artifact_bytes": 99, "status": "approved", "fit_hyperparameters_json": dict(HOOK_BLOCK)}
    o.update(over)
    return {k: v for k, v in o.items() if v is not None}


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _corpus(tmp_path, with_unclassified=False, with_nonintegral=False, with_ruled=False, with_stale=False, with_bucket_pairs=False,
            with_parent_check=False, with_parent_disagreement=False, with_unrecorded_by_name=False, with_parent_nonintegral=False,
            with_statement_check=False):
    """Six objects beside the synthetic seeds: EPOCH (hook-written checkpoint, feb 1 vs recorded 64: CLEARLY WRONG, native carrier), DOSE
    (a hook-written dose copy of it, feb 1 vs 64: CLEARLY WRONG, carried by an adapters bundle), AMBIG (an adapter-filed dose copy, feb
    unstated beside a recorded 8: AMBIGUOUS, listed), FINE (feb 64 = recorded 64: consistent), NONE (feb 32, no recorded value: nothing
    to compare) and, with ``with_unclassified``, ODD (feb 2 vs 64, the hook's signature failing: UNCLASSIFIED, refused by name); with
    ``with_nonintegral``, AUTO (feb 1 beside a recorded ``"auto"``) and HALF (feb 1 beside a recorded 64.5 in the hook's otherwise consistent
    block: truncated to 64 the signature WOULD hold) -- both accepted by the validator (the block is valid JSON; its keys' types are not
    constrained), both UNCLASSIFIED, refused by name with the value and its type. With ``with_ruled``, three
    adapter-filed dose copies as the fit-sidecars adapter emits them since the ruling: RULED (feb 8 = recorded 8, the column marked
    `derived`: CONSISTENT, in neither class), BYNAME (feb NULL beside a recorded 64.0 with the adapter's `unverified` entry naming why --
    the copied record's integral float: AMBIGUOUS, listed as left NULL by name) and DERIVED_ODD (feb 1 vs 64 in the hook's exact shape, but
    the column marked `derived`: NOT the hook's statement, UNCLASSIFIED, refused by name, never written; its parent EPOCH records 64, so the
    parent cross-check lists it too). With ``with_stale``, STALE (feb 8 beside a block in the hook's exact shape -- resolved_batch_size 8,
    8 x 8 = 64 -- and the adapter's `unverified` NULL-by-name entry for the column: a retained earlier statement, which an omitted column
    would have left standing; NOT the hook's statement, UNCLASSIFIED, refused by name, never written). With ``with_bucket_pairs`` (on top
    of ``with_ruled``), a SECOND row in each ambiguous bucket: AMBIG2 beside AMBIG (lora writer, recorded 8, no entry) and BYNAME2 beside
    BYNAME (psr writer, recorded 64.0, `unverified`, a DIFFERENT reason: a copy of another parent) -- two copies a count cannot tell apart,
    which the report must name. With ``with_parent_check``, three derived dose copies for the parent cross-check: AGREE (derived 64, its
    parent FINE's block records 64: agrees), VIACOL (derived 32, its parent NONE records no block value but its column is 32: agrees through
    the parent's column) and ORPHAN (derived 8, its parent NOWHERE is no row of the registry: absent, not a disagreement); with
    ``with_parent_disagreement``, PCHECK (derived 8 = its own block's 8, in neither of the pass's classes, but its parent FINE records 64:
    a parent DISAGREEMENT, listed by name, never written for). With ``with_unrecorded_by_name`` (review round astra, 2026-10-13), three
    dose copies the adapter left NULL by name whose block records NO effective_batch: NOKEY (psr writer, the block without the key), NULLKEY
    (psr writer, the key null in the block) and NOBLOCK (a subtraction's dose child, `lora_subtract.py` writer, no block at all) -- each with
    the adapter's `unverified` entry and reason, AMBIGUOUS (listed as left NULL by name, recorded `-`) -- beside two rows that are NOT in the
    class: UNSAID (a dose copy with NULL, no entry and no block: nothing stated, nothing recorded) and ANCHORISH (derivation
    `lora_subtraction`, an `unverified` entry beside NULL and no block: its own writer's reading). With ``with_parent_nonintegral``
    (review round sol, 2026-10-13), two parents whose RECORDED effective batch is not an integer in its own type -- PFLOAT (column 8, the
    block's `effective_batch` 8.0) and PBOOL (column 1, the block's `true`), each consistent to the pass (its class SQL reads 8 and 8.0, 1
    and true, as equal) -- and their derived dose copies CFLOAT (derived 8) and CBOOL (derived 1): a parent cross-check that `==` alone
    read as agreement, and that is a disagreement of type, listed by name with the value and its type. With ``with_statement_check``
    (review round sol-2, 2026-10-13), four dose copies whose adapter statement does not hold of their OWN row and that the class SQL never
    selects: SFLOAT (derived 8 beside a block recording 8.0 -- SQLite's `8 <> 8.0` is false), SNOKEY (derived 8 beside a block without the
    key: nothing to compare, to the SQL), SNULL (a `derived` entry beside a NULL column, the block recording 8: in the ambiguous class by its
    entry) and SSTALE (the adapter's `unverified` entry beside a standing 8 whose block records 8: the two values agree) -- each with a
    parent that states no batch (or no derived value at all), so the parent cross-check reads none of them as a disagreement either."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    dims.append(synth.rec("object", _epoch_checkpoint(SHA_EPOCH), synth.t(40)))
    dims.append(synth.rec("object", _dose_copy(SHA_AMBIG, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8}), synth.t(41)))
    dims.append(synth.rec("object", _epoch_checkpoint(SHA_FINE, role="epoch_2", fit_effective_batch=64), synth.t(42)))
    dims.append(synth.rec("object", _epoch_checkpoint(SHA_NONE, role="intermediate", fit_effective_batch=32, fit_hyperparameters_json={"lr": 0.001, "epochs": 3}), synth.t(43)))
    if with_unclassified:
        dims.append(synth.rec("object", _epoch_checkpoint(SHA_ODD, role="final", fit_effective_batch=2), synth.t(44)))
    if with_nonintegral:
        dims.append(synth.rec("object", _epoch_checkpoint(SHA_AUTO, role="final", fit_hyperparameters_json={**HOOK_BLOCK, "effective_batch": "auto"}), synth.t(46)))
        dims.append(synth.rec("object", _epoch_checkpoint(SHA_HALF, role="final", fit_hyperparameters_json={**HOOK_BLOCK, "effective_batch": 64.5}), synth.t(47)))
    if with_parent_nonintegral:
        # two consistent parents whose RECORDED effective batch is not an integer in its own type: the pass's own class SQL reads 8 and 8.0
        # (1 and true) as equal, so neither row is selected or listed anywhere but in the parent cross-check of its copy
        dims.append(synth.rec("object", _epoch_checkpoint(SHA_PFLOAT, role="intermediate", fit_effective_batch=8,
                                                          fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8.0}), synth.t(63)))
        dims.append(synth.rec("object", _epoch_checkpoint(SHA_PBOOL, role="intermediate", fit_effective_batch=1,
                                                          fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 1, "effective_batch": True}), synth.t(64)))
    _write(root, "dims", "000001-dims.jsonl", dims)
    bank = [synth.rec("object", _dose_copy(SHA_DOSE, SHA_EPOCH, fit_effective_batch=1), synth.t(45))]
    if with_ruled:
        bank.append(synth.rec("object", _dose_copy(SHA_RULED, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                   fit_effective_batch=8, fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(DERIVED_ENTRY)}), synth.t(48)))
        bank.append(synth.rec("object", _dose_copy(SHA_BYNAME, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_psr_dose.py",
                                                   fit_hyperparameters_json=dict(BYNAME_BLOCK),
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(BYNAME_ENTRY)}), synth.t(49)))
        bank.append(synth.rec("object", _dose_copy(SHA_DERIVED_ODD, SHA_EPOCH, fit_effective_batch=1,
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(DERIVED_ENTRY)}), synth.t(50)))
    if with_stale:
        bank.append(synth.rec("object", _dose_copy(SHA_STALE, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                   fit_effective_batch=8, fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "batch_size": 8, "resolved_batch_size": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(BYNAME_ENTRY)}), synth.t(51)))
    if with_bucket_pairs:
        # a second row in each ambiguous bucket: AMBIG2 shares AMBIG's (writer, recorded, model, no entry); BYNAME2 shares BYNAME's (writer,
        # recorded, model, unverified) with a different reason
        bank.append(synth.rec("object", _dose_copy(SHA_AMBIG2, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8}), synth.t(52)))
        bank.append(synth.rec("object", _dose_copy(SHA_BYNAME2, SHA_FINE, writer="scripts/sdfbase/scale_psr_dose.py",
                                                   fit_hyperparameters_json=dict(BYNAME_BLOCK),
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(BYNAME_ENTRY_2)}), synth.t(53)))
    if with_parent_check:
        bank.append(synth.rec("object", _dose_copy(SHA_AGREE, SHA_FINE, writer="scripts/sdfbase/scale_lora_dose.py", fit_effective_batch=64,
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": derived_entry(SHA_FINE)}), synth.t(54)))
        bank.append(synth.rec("object", _dose_copy(SHA_VIACOL, SHA_NONE, writer="scripts/sdfbase/scale_lora_dose.py", fit_effective_batch=32,
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 32, "effective_batch": 32},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": derived_entry(SHA_NONE)}), synth.t(55)))
        bank.append(synth.rec("object", _dose_copy(SHA_ORPHAN, SHA_NOWHERE, writer="scripts/sdfbase/scale_lora_dose.py", fit_effective_batch=8,
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": derived_entry(SHA_NOWHERE)}), synth.t(56)))
    if with_parent_disagreement:
        bank.append(synth.rec("object", _dose_copy(SHA_PCHECK, SHA_FINE, writer="scripts/sdfbase/scale_lora_dose.py", fit_effective_batch=8,
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": derived_entry(SHA_FINE)}), synth.t(57)))
    if with_parent_nonintegral:
        # derived copies of PFLOAT and PBOOL, each consistent with its own block (in neither of the pass's classes): the cross-check alone reads them
        bank.append(synth.rec("object", _dose_copy(SHA_CFLOAT, SHA_PFLOAT, writer="scripts/sdfbase/scale_lora_dose.py", fit_effective_batch=8,
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": derived_entry(SHA_PFLOAT)}), synth.t(65)))
        bank.append(synth.rec("object", _dose_copy(SHA_CBOOL, SHA_PBOOL, writer="scripts/sdfbase/scale_lora_dose.py", fit_effective_batch=1,
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 1, "effective_batch": 1},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": derived_entry(SHA_PBOOL)}), synth.t(66)))
    if with_unrecorded_by_name:
        nokey_block = {k: v for k, v in HOOK_BLOCK.items() if k != "effective_batch"}
        bank.append(synth.rec("object", _dose_copy(SHA_NOKEY, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_psr_dose.py", fit_hyperparameters_json=nokey_block,
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": unverified_entry(nokey_reason(synth.OBJECT_SHA))}), synth.t(58)))
        bank.append(synth.rec("object", _dose_copy(SHA_NULLKEY, SHA_FINE, writer="scripts/sdfbase/scale_psr_dose.py", fit_hyperparameters_json={**HOOK_BLOCK, "effective_batch": None},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": unverified_entry(nokey_reason(SHA_FINE))}), synth.t(59)))
        bank.append(synth.rec("object", _dose_copy(SHA_NOBLOCK, SHA_FINE, writer="scripts/sdfbase/lora_subtract.py", fit_hyperparameters_json=None,
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": unverified_entry(noblock_reason(SHA_FINE))}), synth.t(60)))
        bank.append(synth.rec("object", _dose_copy(SHA_UNSAID, SHA_FINE, writer="scripts/sdfbase/scale_lora_dose.py", fit_hyperparameters_json=None), synth.t(61)))
        bank.append(synth.rec("object", _dose_copy(SHA_ANCHORISH, SHA_EPOCH, writer="scripts/sdfbase/lora_subtract.py", derivation="lora_subtraction",
                                                   parent_object_ids_json=[SHA_EPOCH, SHA_FINE], base_scale=None, fit_hyperparameters_json=None,
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": unverified_entry(ANCHORISH_REASON)}), synth.t(62)))
    if with_statement_check:
        # four copies the class SQL never selects (8 = 8.0 to SQLite; a missing key is nothing to compare; a NULL column; two equal 8s) whose
        # adapter statement is nonetheless false of their own row; their parent (the seeds' fitted object) states no batch, so the parent
        # cross-check is clean and the statement check is the one thing that reads them
        bank.append(synth.rec("object", _dose_copy(SHA_SFLOAT, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                   fit_effective_batch=8, fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8.0},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(DERIVED_ENTRY)}), synth.t(67)))
        bank.append(synth.rec("object", _dose_copy(SHA_SNOKEY, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                   fit_effective_batch=8, fit_hyperparameters_json={k: v for k, v in HOOK_BLOCK.items() if k != "effective_batch"},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(DERIVED_ENTRY)}), synth.t(68)))
        bank.append(synth.rec("object", _dose_copy(SHA_SNULL, synth.OBJECT_SHA, writer="scripts/sdfbase/scale_psr_dose.py",
                                                   fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(DERIVED_ENTRY)}), synth.t(69)))
        bank.append(synth.rec("object", _dose_copy(SHA_SSTALE, SHA_FINE, writer="scripts/sdfbase/scale_lora_dose.py", object_kind="lora", paper_label="control_lora",
                                                   fit_effective_batch=8, fit_hyperparameters_json={**HOOK_BLOCK, "accum_steps": 8, "effective_batch": 8},
                                                   field_confidence_json={**OTHER_ENTRY, "fit_effective_batch": dict(BYNAME_ENTRY_2)}), synth.t(70)))
    _write(root, os.path.join("adapters", "objects_bank"), "000001-object.jsonl", bank)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return root


def _live(root, sha):
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT fit_effective_batch, fit_hyperparameters_json, field_confidence_json, extra_json, status, field_versions_json, named_role, "
                       "fit_layers_json, parent_object_ids_json FROM object WHERE object_id = ?", (sha,)).fetchone()
    conn.close()
    assert row is not None, sha
    feb, hp, conf, extra, status, fv, role, layers, parents = row
    return {"feb": feb, "hp": json.loads(hp) if hp else None, "conf": json.loads(conf or "{}"), "extra": json.loads(extra or "{}"), "status": status,
            "fv": json.loads(fv or "{}"), "named_role": role, "layers": layers, "parents": parents}


def _read_one(path):
    (_i, rec, _s), = list(R.read_record_file(path, compute_sha=False))
    return rec


def _just_after(written_utc):
    """A stamp one millisecond after ``written_utc``: a record that must outrank one the pass wrote with the wall clock is stamped from
    THAT record's stamp, never from the synthetic clock (whose stamps lie months before the wall clock and would lose the fold)."""
    return R.epoch_ms_to_utc(R.utc_to_epoch_ms(written_utc) + 1)


def _rebuild_equal(root, tmp_path):
    live = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
    assert not diff, diff
    live.close()
    rebuilt.close()


def _tree(root):
    out = {}
    for dp, _dns, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(("-wal", "-shm")):
                continue
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, root)] = (os.path.getsize(p), os.path.getmtime(p))
    return out


# --------------------------------------------------------------------------------------------
# the pure predicate
# --------------------------------------------------------------------------------------------


def test_the_hook_signature_is_the_predicate():
    assert RC.batch_hook_signature(1, 64, 1, 1, 64)                        # the live class: micro 1, accum 64, recorded 64, stated 1
    assert RC.batch_hook_signature(8, 64, 8, 8, 8)                         # micro 8, accum 8, recorded 64, stated 8: the signature holds too
    assert RC.batch_hook_signature(8, 64, 8, 8, 4) is False                # the block's own product (8 x 4) is not its effective_batch (64)
    assert RC.batch_hook_signature(2, 64, 1, 1, 64) is False               # the stated value is not the block's micro-batch
    assert RC.batch_hook_signature(64, 64, 1, 1, 64) is False              # agreeing values are never in the class
    assert RC.batch_hook_signature(1, 64, None, 1, 64) is False and RC.batch_hook_signature(None, 64, 1, 1, 64) is False
    # every batch value in the record's own type: a float (an integral one too), a string, a boolean make the row unclassified; nothing coerced
    assert RC.batch_hook_signature(1, 64.5, 1, 1, 64) is False             # int(64.5) = 64 WOULD satisfy the signature: never truncated
    assert RC.batch_hook_signature(1, "auto", 1, 1, 64) is False and RC.batch_hook_signature(1, "64", 1, 1, 64) is False
    assert RC.batch_hook_signature(1, 64.0, 1, 1, 64) is False and RC.batch_hook_signature(1, 64, 1, 1, 64.0) is False
    assert RC.batch_hook_signature(1, 64, 1, True, 64) is False and RC.batch_hook_signature(True, 64, 1, 1, 64) is False
    assert RC.integral_batch(64) and RC.integral_batch(0)
    assert not any(RC.integral_batch(v) for v in (64.0, 64.5, "64", "auto", True, False, None, [64]))
    # the reason names the first failing value with its type, or the signature's failing condition; None for the class
    assert RC.batch_unclassified_reason(1, 64, 1, 1, 64) is None
    assert RC.batch_unclassified_reason(1, 64.5, 1, 1, 64) == HALF_REASON
    assert RC.batch_unclassified_reason(1, "auto", 1, 1, 64) == AUTO_REASON
    assert RC.batch_unclassified_reason(1, 64, 1, True, 64) == "fit_hyperparameters_json.batch_size is True (bool), not an integer; nothing is coerced"
    assert RC.batch_unclassified_reason(1, 64, 1, 1, None) == "fit_hyperparameters_json.accum_steps is not stated in the row's record; nothing is coerced"
    assert RC.batch_unclassified_reason(1.0, 64, 1, 1, 64) == "fit_effective_batch is 1.0 (float), not an integer; nothing is coerced"
    assert RC.batch_unclassified_reason(64, 64, 1, 1, 64) == "the two values agree"
    assert RC.batch_unclassified_reason(2, 64, 1, 1, 64) == ODD_REASON
    assert RC.batch_unclassified_reason(8, 64, 8, 8, 4) == ("the block's own batch_size 8 x accum_steps 4 = 32 is not its effective_batch 64: the sidecar "
                                                            "hook's signature does not hold (the trainer's identity fails in the row's record)")


# --------------------------------------------------------------------------------------------
# the pass on a temp root
# --------------------------------------------------------------------------------------------


def test_dry_run_classes_the_rows_lists_the_ambiguous_ones_and_writes_nothing(tmp_path, capsys):
    root = _corpus(tmp_path, with_unclassified=True)
    rep = RC.reconcile_fit_effective_batch(root)
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and [r.object_id for r in rep.unclassified] == [SHA_ODD]
    assert not rep.applied and rep.written == [] and rep.held == []
    epoch = next(r for r in rep.rows if r.object_id == SHA_EPOCH)
    assert (epoch.live, epoch.recorded, epoch.clearly_wrong, epoch.carrier, epoch.group) == (1, 64, True, "native", ("relu_steer", "epoch_1", "native"))
    assert set(epoch.row) == set(c for c in R.REQUIRED_FIELDS["object"] if c != "status") | {"fit_corpus_id", "fit_corpus_sha256", "fit_model_id", "fit_layers_json"}
    assert epoch.row["fit_layers_json"] == [4, 5, 6] and epoch.confidence == OTHER_ENTRY and epoch.hyper == HOOK_BLOCK
    dose = next(r for r in rep.rows if r.object_id == SHA_DOSE)
    assert (dose.carrier, dose.group, dose.row["parent_object_ids_json"]) == ("adapter", ("relu_steer", "base_scale 0.75", "adapter"), [SHA_EPOCH])
    assert "fit_corpus_id" not in dose.row and "parent_object_ids_json" in dose.row
    assert rep.by_group() == {("relu_steer", "base_scale 0.75", "adapter"): (1, 1, 64), ("relu_steer", "epoch_1", "native"): (1, 1, 64)}
    # the ambiguous group carries the row's own confidence for the column (None = no entry, the adapter's pre-ruling emission) and names
    # its one row: the object id, no reason (a row with no entry states none)
    assert [g.summary for g in rep.ambiguous] == [("scripts/sdfbase/scale_lora_dose.py", 8, synth.FIT_MODEL_ID, None, 1)] and rep.nothing_to_compare == 1
    assert rep.ambiguous[0].members == [(SHA_AMBIG, None)] and rep.ambiguous[0].object_ids == [SHA_AMBIG]
    d = rep.as_dict()
    assert d["rows"] == 2 and d["unclassified"] == 1 and d["ambiguous_rows"] == 1 and d["nothing_to_compare"] == 1 and d["written"] == 0
    assert d["ambiguous"] == [{"fit_writer": "scripts/sdfbase/scale_lora_dose.py", "recorded": 8, "fit_model_id": synth.FIT_MODEL_ID, "stated": None,
                               "statement": RC.BATCH_AMBIGUOUS_UNSTATED, "rows": 1, "objects": [{"object_id": SHA_AMBIG, "unresolved": None}]}]
    # the CLI's dry run: the classes, exit 0, nothing written under the root (no last_run stamp)
    before = _tree(root)
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    out = capsys.readouterr().out
    assert f"  {'relu_steer':<16} {'epoch_1':<16} {'native':<8} 1 -> 64 {1:>6}" in out and f"  {'relu_steer':<16} {'base_scale 0.75':<16} {'adapter':<8} 1 -> 64 {1:>6}" in out
    assert ("ambiguous (fit_effective_batch NULL beside a recorded effective_batch, or a dose copy's NULL stated by the adapter's own entry with none recorded: "
            "a dose copy the fit-sidecars adapter has not restated yet, or left NULL "
            "by name -- its re-emission is the correction, the user 2026-10-13; listed by object id, not corrected): 1 row(s)") in out
    assert f"  {'scripts/sdfbase/scale_lora_dose.py':<40} {'8':<4} {synth.FIT_MODEL_ID:<20} {1:>6}  {RC.BATCH_AMBIGUOUS_UNSTATED}\n    {SHA_AMBIG}\n" in out
    assert "nothing to compare (fit_effective_batch set, no recorded effective_batch): 1 row(s)" in out
    assert f"{UNCLASSIFIED_HEAD}: 1 row(s)\n  {SHA_ODD} 2 -> 64 [relu_steer]: {ODD_REASON}" in out
    assert "rows 2 in 2 (kind, role, carrier) group(s); nothing written (dry run)" in out and '"ambiguous_rows": 1' in out
    assert _tree(root) == before and not os.path.exists(os.path.join(root, "state", "last_run"))


def test_apply_writes_one_object_record_per_clearly_wrong_row_that_folds_to_the_recorded_batch_and_leaves_every_other_column(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root = _corpus(tmp_path)
    before = {sha: _live(root, sha) for sha in (SHA_EPOCH, SHA_DOSE, SHA_AMBIG, SHA_FINE, SHA_NONE)}
    assert before[SHA_EPOCH]["feb"] == 1 and before[SHA_DOSE]["feb"] == 1 and before[SHA_AMBIG]["feb"] is None
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert app.applied and len(app.written) == 2 and app.held == []
    epoch = next(r for r in app.rows if r.object_id == SHA_EPOCH)
    assert epoch.written == os.path.join(root, "records", "registry_owner", f"object__{SHA_EPOCH[:16]}", "000001-object.jsonl")
    rec = _read_one(epoch.written)
    assert rec["record_type"] == "object" and rec["producer"]["script"] == RC.BATCH_PRODUCER_SCRIPT
    p = rec["payload"]
    assert p["object_id"] == SHA_EPOCH and p["fit_effective_batch"] == 64 and "status" not in p and "fit_hyperparameters_json" not in p and "named_role" not in p
    assert p["fit_target_kind"] == "corpus_document_fit" and p["fit_corpus_id"] == "synthetic-corpus" and p["fit_layers_json"] == [4, 5, 6] and p["fit_model_id"] == synth.FIT_MODEL_ID
    assert p["field_confidence_json"] == {**OTHER_ENTRY, "fit_effective_batch": {"confidence": "recorded", "source": p["field_confidence_json"]["fit_effective_batch"]["source"]}}
    assert "batch_size x accum_steps" in p["field_confidence_json"]["fit_effective_batch"]["source"] and "resolved_batch_size" in p["field_confidence_json"]["fit_effective_batch"]["source"]
    assert p["extra"] == {RC.BATCH_MARKER_KEY: {"replaced": 1, "derived_from": {"effective_batch": 64, "batch_size": 1, "accum_steps": 64, "resolved_batch_size": 1},
                                                "pass": RC.BATCH_PRODUCER_SCRIPT}}
    assert isinstance(VAL.validate_record(rec, epoch.written, R.record_sha(rec)), VAL.Validated)
    dose_rec = _read_one(next(r for r in app.rows if r.object_id == SHA_DOSE).written)["payload"]
    assert dose_rec["fit_target_kind"] == "derived" and dose_rec["parent_object_ids_json"] == [SHA_EPOCH] and "fit_corpus_id" not in dose_rec and dose_rec["fit_effective_batch"] == 64
    # before the ingest a second apply holds both (their records are in the log, not folded); the CLI exits 2 and names the hold
    again = RC.reconcile_fit_effective_batch(root, apply=True)
    assert again.written == [] and all(r.pending_own and "this pass's own correction record" in r.refused for r in again.rows)
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 2
    assert "2 row(s) held" in capsys.readouterr().err
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"  {SHA_EPOCH} 1 -> 64 [relu_steer]" in out and "inconsistent 2 row(s)" in out
    # the fold: the column moves to 64 on both; status, the hyperparameter block (value AND stamp), the role stand; the confidence JSON keeps
    # the standing entry beside the new one; the marker is in extra; the other three rows are untouched
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    for sha in (SHA_EPOCH, SHA_DOSE):
        after = _live(root, sha)
        assert after["feb"] == 64 and after["status"] == before[sha]["status"] and after["hp"] == HOOK_BLOCK and after["named_role"] == before[sha]["named_role"]
        assert after["fv"]["fit_hyperparameters_json"] == before[sha]["fv"]["fit_hyperparameters_json"] and after["fv"]["status"] == before[sha]["fv"]["status"]
        assert after["fv"]["fit_effective_batch"] != before[sha]["fv"]["fit_effective_batch"]
        assert after["conf"]["fit_effective_batch"]["confidence"] == "recorded" and after["extra"][RC.BATCH_MARKER_KEY]["replaced"] == 1
        assert after["layers"] == before[sha]["layers"] and after["parents"] == before[sha]["parents"]
    assert _live(root, SHA_EPOCH)["conf"]["fit_thinking_render"] == OTHER_ENTRY["fit_thinking_render"]
    for sha in (SHA_AMBIG, SHA_FINE, SHA_NONE):
        assert _live(root, sha) == before[sha]
    # the verify reads zero; the ambiguous row still stands unstated; idempotent: a second apply selects nothing, the CLI exits 0
    ver = RC.verify_fit_effective_batch(root)
    assert ver.rows == [] and RC.format_batch_report(ver).endswith("inconsistent 0 row(s); the class is consistent")
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 0
    third = RC.reconcile_fit_effective_batch(root, apply=True)
    assert third.rows == [] and third.written == [] and [g.summary for g in third.ambiguous] == [("scripts/sdfbase/scale_lora_dose.py", 8, synth.FIT_MODEL_ID, None, 1)]
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 0
    assert sorted(os.listdir(os.path.join(root, "records", "registry_owner", f"object__{SHA_EPOCH[:16]}"))) == ["000001-object.jsonl"]
    _rebuild_equal(root, tmp_path)


def test_an_unclassified_disagreement_is_refused_by_name_and_never_written(tmp_path, capsys):
    root = _corpus(tmp_path, with_unclassified=True)
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    odd = next(r for r in app.rows if r.object_id == SHA_ODD)
    assert odd.written is None and "unclassified disagreement, refused by name" in odd.refused and "signature does not hold" in odd.refused
    assert len(app.written) == 2 and not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{SHA_ODD[:16]}"))
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 2
    err = capsys.readouterr().err
    assert "held" in err and "unclassified" in err
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    # after the fold the clearly-wrong rows left the class; the unclassified row still stands and the verify names it, exit 3
    assert _live(root, SHA_ODD)["feb"] == 2 and _live(root, SHA_EPOCH)["feb"] == 64
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    assert f"  {SHA_ODD} 2 -> 64 [relu_steer, unclassified: {ODD_REASON}]" in capsys.readouterr().out


def test_the_writer_adds_no_central_stamp_even_with_an_overrides_file_in_the_environment(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    f = tmp_path / "operator_shell.env"
    f.write_text("OVERRIDE_ID=operator_shell_v9\n")
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    assert emit.run_overrides_stamp()["id"] == "operator_shell_v9"
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    rec = _read_one(next(r for r in app.rows if r.object_id == SHA_EPOCH).written)
    assert set(rec["payload"]["extra"]) == {RC.BATCH_MARKER_KEY}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    assert "run_overrides" not in _live(root, SHA_EPOCH)["extra"]


def test_a_row_is_held_behind_an_unfolded_object_record_wherever_in_the_log_it_lies_and_the_target_directory_is_read_again_at_the_write(tmp_path, monkeypatch, capsys):
    """A native re-emission of the object (in its producer's own directory, not yet folded) holds the row: the dry run names it, the apply
    writes nothing for it (exit 2); after the ingest the predicate decides again from the folded row. A record that lands in the target
    directory after the walk (the walk stubbed to have seen nothing) holds it at the write."""
    root = _corpus(tmp_path)
    re_emission = synth.rec("object", _epoch_checkpoint(SHA_EPOCH), synth.t(200))
    _write(root, os.path.join(synth.CAMPAIGN, f"object__{SHA_EPOCH[:16]}"), "000001-object.jsonl", [re_emission])
    rep = RC.reconcile_fit_effective_batch(root)
    held = {r.object_id: r for r in rep.selected if r.pending}
    assert set(held) == {SHA_EPOCH} and held[SHA_EPOCH].pending == R.record_sha(re_emission) and not held[SHA_EPOCH].pending_own
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert "pending (an object record of the row is in the log, not folded yet): 1 row(s)" in out and f"  {SHA_EPOCH} record {R.record_sha(re_emission)[:12]}" in out
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    by = {r.object_id: r for r in app.rows}
    assert by[SHA_EPOCH].written is None and "an object record of this object is in the log and not folded yet" in by[SHA_EPOCH].refused and by[SHA_DOSE].written
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 2
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    # the re-emission restated the hook's 1: the row is still in the class and is corrected now; the dose copy left it
    assert _live(root, SHA_EPOCH)["feb"] == 1 and _live(root, SHA_DOSE)["feb"] == 64
    app2 = RC.reconcile_fit_effective_batch(root, apply=True)
    assert [r.object_id for r in app2.rows] == [SHA_EPOCH] and app2.written
    # the late read: a record planted in the target directory after the (stubbed) walk holds the row at the write. The producer's
    # re-emission must OUTRANK the correction the pass just wrote with the wall clock, so it is stamped one millisecond after that record
    # (a synthetic-clock stamp would lose the fold and the row would never re-enter the class); the row is then asserted back in the class
    rep3 = H.ingest_once(root, workers=0, min_free=0)
    assert rep3.refused == 0, rep3.refusals
    assert _live(root, SHA_EPOCH)["feb"] == 64
    corrected_at = _read_one(app2.written[0])["written_utc"]
    re_emitted_at = _just_after(corrected_at)
    _write(root, os.path.join(synth.CAMPAIGN, f"object__{SHA_EPOCH[:16]}"), "000002-object.jsonl", [synth.rec("object", _epoch_checkpoint(SHA_EPOCH), re_emitted_at)])
    rep4 = H.ingest_once(root, workers=0, min_free=0)
    assert rep4.refused == 0, rep4.refusals
    assert _live(root, SHA_EPOCH)["feb"] == 1 and _live(root, SHA_EPOCH)["fv"]["fit_effective_batch"][0] == re_emitted_at    # [written_utc, sha]
    assert [r.object_id for r in RC.reconcile_fit_effective_batch(root).selected] == [SHA_EPOCH]          # back in the class
    late = synth.rec("object", _epoch_checkpoint(SHA_EPOCH), _just_after(re_emitted_at))
    _write(root, os.path.join("registry_owner", f"object__{SHA_EPOCH[:16]}"), "000003-object.jsonl", [late])
    monkeypatch.setattr(RC, "object_statements", lambda records_root, object_ids=None: {})
    app3 = RC.reconcile_fit_effective_batch(root, apply=True)
    (row,) = app3.rows
    assert row.object_id == SHA_EPOCH and row.written is None and row.pending == R.record_sha(late) and not row.pending_own
    assert row.refused == f"an object record of this object is in the log and not folded yet ({R.record_sha(late)[:12]}); ingest first, then re-run"


def test_a_recorded_batch_that_is_not_an_integer_is_unclassified_refused_by_name_and_never_truncated_or_dropped(tmp_path, capsys):
    """The validator constrains `fit_hyperparameters_json` to valid JSON, not its keys' types, so an accepted row may record
    `effective_batch` as a string ("auto") or a float (64.5). Neither is coerced: both are listed UNCLASSIFIED with the value and its type
    (a pass coercing with int() would drop the "auto" row from the report and read a clean class past it, and would truncate 64.5 to 64,
    satisfy the hook's signature and WRITE a correction to a value the record never stated); the apply writes nothing for them (exit 2);
    after the fold both still stand, their recorded value as stated, named by the verify (exit 3). The clearly-wrong rows beside them are
    corrected as before."""
    root = _corpus(tmp_path, with_nonintegral=True)
    rep = RC.reconcile_fit_effective_batch(root)
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and [r.object_id for r in rep.unclassified] == sorted([SHA_AUTO, SHA_HALF])
    by = {r.object_id: r for r in rep.rows}
    assert (by[SHA_AUTO].live, by[SHA_AUTO].recorded, by[SHA_AUTO].clearly_wrong, by[SHA_AUTO].unclassified) == (1, "auto", False, AUTO_REASON)
    assert (by[SHA_HALF].live, by[SHA_HALF].recorded, by[SHA_HALF].clearly_wrong, by[SHA_HALF].unclassified) == (1, 64.5, False, HALF_REASON)
    assert by[SHA_EPOCH].unclassified is None and by[SHA_HALF].hyper["effective_batch"] == 64.5
    d = rep.as_dict()
    assert d["rows"] == 2 and d["unclassified"] == 2 and {c["unclassified"] for c in d["corrections"] if not c["clearly_wrong"]} == {AUTO_REASON, HALF_REASON}
    assert rep.by_group() == {("relu_steer", "base_scale 0.75", "adapter"): (1, 1, 64), ("relu_steer", "epoch_1", "native"): (1, 1, 64)}
    # the CLI's dry run names both with the value in its own type (the string quoted, the float as written)
    before = _tree(root)
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert f"{UNCLASSIFIED_HEAD}: 2 row(s)" in out
    assert f"  {SHA_AUTO} 1 -> 'auto' [relu_steer]: {AUTO_REASON}" in out and f"  {SHA_HALF} 1 -> 64.5 [relu_steer]: {HALF_REASON}" in out
    assert "rows 2 in 2 (kind, role, carrier) group(s); nothing written (dry run)" in out and _tree(root) == before
    # the apply: nothing written for either, no target directory made, the refusal names the value and its type; the two clearly-wrong rows written
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert len(app.written) == 2 and {r.object_id for r in app.held} == {SHA_AUTO, SHA_HALF}
    for sha, reason in ((SHA_AUTO, AUTO_REASON), (SHA_HALF, HALF_REASON)):
        r = next(x for x in app.rows if x.object_id == sha)
        assert r.written is None and r.refused.endswith(f"and {reason}: an unclassified disagreement, refused by name, never corrected by this pass")
        assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{sha[:16]}"))
    assert next(x for x in app.rows if x.object_id == SHA_HALF).refused.startswith("fit_effective_batch 1 disagrees with the recorded effective_batch 64.5 and ")
    # the CLI's apply after it: the two written rows are held behind their own records, the two unclassified rows refused: 4 held, exit 2
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 2
    assert "4 row(s) held" in capsys.readouterr().err
    # the fold: the clearly-wrong rows move to 64; the two non-integer rows keep the hook's 1 AND their recorded value as stated, no marker
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert _live(root, SHA_EPOCH)["feb"] == 64 and _live(root, SHA_DOSE)["feb"] == 64
    for sha, recorded in ((SHA_AUTO, "auto"), (SHA_HALF, 64.5)):
        after = _live(root, sha)
        assert after["feb"] == 1 and after["hp"]["effective_batch"] == recorded and RC.BATCH_MARKER_KEY not in after["extra"] and "fit_effective_batch" not in after["conf"]
    # the verify names both with the reason, exit 3; the class itself is empty; a second apply writes nothing and still refuses them by name
    ver = RC.verify_fit_effective_batch(root)
    assert ver.selected == [] and [r.object_id for r in ver.listed] == sorted([SHA_AUTO, SHA_HALF])
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"  {SHA_AUTO} 1 -> 'auto' [relu_steer, unclassified: {AUTO_REASON}]" in out and f"  {SHA_HALF} 1 -> 64.5 [relu_steer, unclassified: {HALF_REASON}]" in out
    assert "inconsistent 2 row(s)" in out
    again = RC.reconcile_fit_effective_batch(root, apply=True)
    assert again.written == [] and again.selected == [] and {r.object_id for r in again.held} == {SHA_AUTO, SHA_HALF}
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply", "--json"]) == 2
    captured = capsys.readouterr()
    assert "2 row(s) held" in captured.err
    assert {c["object_id"]: c["unclassified"] for c in json.loads(captured.out)["corrections"]} == {SHA_AUTO: AUTO_REASON, SHA_HALF: HALF_REASON}
    assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{SHA_AUTO[:16]}"))
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the ruling on the dose copies: the adapter's statement, not a correction
# --------------------------------------------------------------------------------------------


DERIVED_ODD_REASON_HEAD = "fit_effective_batch is a derived statement (field_confidence_json: derived -- "


def test_a_dose_copy_the_adapter_restated_is_consistent_one_left_null_by_name_is_listed_as_such_and_a_derived_disagreement_is_refused(tmp_path, capsys):
    """The fit-sidecars adapter states a dose copy's `fit_effective_batch` as its parent's, `derived`, and its re-emission
    is the correction -- this pass writes nothing for a dose copy. So RULED (feb 8 = recorded 8, marked derived) is in neither class; the
    ambiguous class tells a row the adapter has not restated (AMBIG: no confidence entry for the column) from one it left NULL by name
    (BYNAME: the adapter's `unverified` entry, the reason in the row, the block's 64.0 as stored), each by its own statement in the report and the JSON; and a row whose
    column the adapter marked `derived` but whose values disagree in the hook's exact shape (DERIVED_ODD) is NOT the hook's statement: it is
    unclassified with the derived reason, refused by name, never written, and the verify names it (exit 3) while the two hook rows beside it
    are corrected as before."""
    root = _corpus(tmp_path, with_ruled=True)
    rep = RC.reconcile_fit_effective_batch(root)
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and [r.object_id for r in rep.unclassified] == [SHA_DERIVED_ODD]
    assert SHA_RULED not in {r.object_id for r in rep.rows}
    odd = next(r for r in rep.rows if r.object_id == SHA_DERIVED_ODD)
    assert (odd.live, odd.recorded, odd.clearly_wrong) == (1, 64, False) and odd.unclassified.startswith(DERIVED_ODD_REASON_HEAD)
    assert "the parent's effective batch" in odd.unclassified and odd.unclassified.endswith("not the sidecar hook's reading; the hook's signature does not apply")
    assert RC.batch_derived_reason({"fit_effective_batch": {"confidence": "recorded", "source": "x"}}) is None and RC.batch_derived_reason({}) is None
    assert RC.batch_derived_reason({"fit_effective_batch": dict(BYNAME_ENTRY)}).startswith(STALE_REASON_HEAD)       # the adapter's NULL-by-name entry
    # the ambiguous class: the pre-ruling row (no entry) and the by-name row (the adapter's unverified entry), told apart, each named with
    # its row's own reason; RULED is in neither
    assert [g.summary for g in rep.ambiguous] == [("scripts/sdfbase/scale_lora_dose.py", 8, synth.FIT_MODEL_ID, None, 1),
                                                   ("scripts/sdfbase/scale_psr_dose.py", 64.0, synth.FIT_MODEL_ID, "unverified", 1)]
    assert [g.members for g in rep.ambiguous] == [[(SHA_AMBIG, None)], [(SHA_BYNAME, BYNAME_REASON)]]
    assert isinstance(rep.ambiguous[1].recorded, float)                 # the block's 64.0 as stored, never coerced
    assert [a["statement"] for a in rep.as_dict()["ambiguous"]] == [RC.BATCH_AMBIGUOUS_UNSTATED, RC.BATCH_AMBIGUOUS_BY_NAME]
    assert [a["objects"] for a in rep.as_dict()["ambiguous"]] == [[{"object_id": SHA_AMBIG, "unresolved": None}], [{"object_id": SHA_BYNAME, "unresolved": BYNAME_REASON}]]
    assert RC.batch_ambiguous_statement("derived") == "confidence entry 'derived' beside a NULL column (read the row)"
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert f"  {'scripts/sdfbase/scale_lora_dose.py':<40} {'8':<4} {synth.FIT_MODEL_ID:<20} {1:>6}  {RC.BATCH_AMBIGUOUS_UNSTATED}\n    {SHA_AMBIG}\n" in out
    assert f"  {'scripts/sdfbase/scale_psr_dose.py':<40} {'64.0':<4} {synth.FIT_MODEL_ID:<20} {1:>6}  {RC.BATCH_AMBIGUOUS_BY_NAME}\n    {SHA_BYNAME}  {BYNAME_REASON}\n" in out
    assert f"{UNCLASSIFIED_HEAD}: 1 row(s)\n  {SHA_DERIVED_ODD} 1 -> 64 [relu_steer]: {DERIVED_ODD_REASON_HEAD}" in out
    assert "rows 2 in 2 (kind, role, carrier) group(s); nothing written (dry run)" in out
    # the apply: the two hook rows written, the derived disagreement refused by name with nothing written; RULED and BYNAME untouched
    before = {sha: _live(root, sha) for sha in (SHA_RULED, SHA_BYNAME, SHA_DERIVED_ODD)}
    assert before[SHA_RULED]["feb"] == 8 and before[SHA_BYNAME]["feb"] is None and before[SHA_BYNAME]["conf"]["fit_effective_batch"] == BYNAME_ENTRY
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert len(app.written) == 2 and {r.object_id for r in app.held} == {SHA_DERIVED_ODD}
    refused = next(r for r in app.rows if r.object_id == SHA_DERIVED_ODD)
    assert refused.written is None and DERIVED_ODD_REASON_HEAD in refused.refused and refused.refused.endswith("refused by name, never corrected by this pass")
    assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{SHA_DERIVED_ODD[:16]}"))
    assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{SHA_RULED[:16]}"))
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert _live(root, SHA_EPOCH)["feb"] == 64 and _live(root, SHA_DOSE)["feb"] == 64
    for sha in (SHA_RULED, SHA_BYNAME, SHA_DERIVED_ODD):
        assert _live(root, sha) == before[sha]
    assert _live(root, SHA_RULED)["conf"]["fit_effective_batch"] == DERIVED_ENTRY
    # the verify: the derived disagreement still stands, named with its reason (exit 3); RULED and BYNAME are not listed
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"  {SHA_DERIVED_ODD} 1 -> 64 [relu_steer, unclassified: {DERIVED_ODD_REASON_HEAD}" in out and "inconsistent 1 row(s)" in out
    assert SHA_RULED not in out and SHA_BYNAME not in out
    _rebuild_equal(root, tmp_path)


def test_a_value_standing_beside_the_adapters_null_by_name_entry_is_refused_never_corrected(tmp_path, capsys):
    """The pass's backstop for a retained value. STALE carries 8 beside a block in the hook's exact shape (resolved_batch_size 8,
    8 x 8 = 64) and the adapter's `unverified` entry for the column: the fold a re-emission that OMITTED the column would have left (the
    adapter states null now, so the shape should not fold again; a row of it may still stand). The five values alone read as the hook's
    signature, but the entry says the adapter states no value for the column, so the 8 is a retained earlier statement, not the hook's
    reading: unclassified with that reason, refused by name on --apply, never written, named by the verify (exit 3) while the two hook rows
    beside it are corrected as before."""
    root = _corpus(tmp_path, with_stale=True)
    rep = RC.reconcile_fit_effective_batch(root)
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and [r.object_id for r in rep.unclassified] == [SHA_STALE]
    stale = next(r for r in rep.rows if r.object_id == SHA_STALE)
    assert (stale.live, stale.recorded, stale.clearly_wrong) == (8, 64, False) and stale.unclassified.startswith(STALE_REASON_HEAD)
    assert BYNAME_REASON[:60] in stale.unclassified      # the entry's reason, named (its head: the pass truncates the text as the derived branch does)
    assert stale.unclassified.endswith("a retained earlier statement, not the sidecar hook's reading; the hook's signature does not apply")
    assert RC.batch_hook_signature(8, 64, 8, 8, 8)              # the five values alone WOULD read as the hook's: the row's own entry is what refuses it
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert f"{UNCLASSIFIED_HEAD}: 1 row(s)\n  {SHA_STALE} 8 -> 64 [lora]: {STALE_REASON_HEAD}" in out
    before = _live(root, SHA_STALE)
    assert before["feb"] == 8 and before["conf"]["fit_effective_batch"] == BYNAME_ENTRY
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert len(app.written) == 2 and {r.object_id for r in app.held} == {SHA_STALE}
    refused = next(r for r in app.rows if r.object_id == SHA_STALE)
    assert refused.written is None and STALE_REASON_HEAD in refused.refused and refused.refused.endswith("refused by name, never corrected by this pass")
    assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{SHA_STALE[:16]}"))
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert _live(root, SHA_STALE) == before and _live(root, SHA_EPOCH)["feb"] == 64 and _live(root, SHA_DOSE)["feb"] == 64
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"  {SHA_STALE} 8 -> 64 [lora, unclassified: {STALE_REASON_HEAD}" in out and "inconsistent 1 row(s)" in out
    _rebuild_equal(root, tmp_path)


def test_every_ambiguous_row_is_listed_by_object_id_with_its_reason_when_several_copies_share_a_bucket(tmp_path, capsys):
    """Two copies in each ambiguous bucket -- AMBIG and AMBIG2 (lora writer, recorded 8, no entry), BYNAME and BYNAME2 (psr writer,
    recorded 64.0, `unverified`, two DIFFERENT reasons: copies of two parents, each named). A count per bucket said 2 and nothing else, so
    an operator could not tell which copy's copied record to repair; the report names every row under its group with the adapter's own
    reason, in the text and the JSON alike, the aggregate beside them, and the apply and the verify leave every one of them as it was."""
    root = _corpus(tmp_path, with_ruled=True, with_bucket_pairs=True)
    rep = RC.reconcile_fit_effective_batch(root)
    assert [g.summary for g in rep.ambiguous] == [("scripts/sdfbase/scale_lora_dose.py", 8, synth.FIT_MODEL_ID, None, 2),
                                                   ("scripts/sdfbase/scale_psr_dose.py", 64.0, synth.FIT_MODEL_ID, "unverified", 2)]
    unstated, byname = rep.ambiguous
    unstated_rows = sorted([(SHA_AMBIG, None), (SHA_AMBIG2, None)])
    byname_rows = sorted([(SHA_BYNAME, BYNAME_REASON), (SHA_BYNAME2, BYNAME_REASON_2)])
    assert unstated.members == unstated_rows and unstated.object_ids == [oid for oid, _w in unstated_rows] and unstated.rows == 2
    assert byname.members == byname_rows and byname.rows == 2
    assert (unstated.statement, byname.statement) == (RC.BATCH_AMBIGUOUS_UNSTATED, RC.BATCH_AMBIGUOUS_BY_NAME)
    d = rep.as_dict()
    assert d["ambiguous_rows"] == 4 and [a["rows"] for a in d["ambiguous"]] == [2, 2]
    assert d["ambiguous"][0]["objects"] == [{"object_id": oid, "unresolved": None} for oid, _w in unstated_rows]
    assert d["ambiguous"][1]["objects"] == [{"object_id": oid, "unresolved": why} for oid, why in byname_rows]
    # the text: the group's line (the count kept), then one line per row -- the id, and the reason where the row carries one
    text = RC.format_batch_report(rep)
    assert "listed by object id, not corrected): 4 row(s)" in text
    lora_line = f"  {'scripts/sdfbase/scale_lora_dose.py':<40} {'8':<4} {synth.FIT_MODEL_ID:<20} {2:>6}  {RC.BATCH_AMBIGUOUS_UNSTATED}"
    psr_line = f"  {'scripts/sdfbase/scale_psr_dose.py':<40} {'64.0':<4} {synth.FIT_MODEL_ID:<20} {2:>6}  {RC.BATCH_AMBIGUOUS_BY_NAME}"
    assert (lora_line + "".join(f"\n    {oid}" for oid, _w in unstated_rows) + "\n" + psr_line
            + "".join(f"\n    {oid}  {why}" for oid, why in byname_rows) + "\nnothing to compare") in text
    # the CLI's text and JSON carry the same names
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    for sha in (SHA_AMBIG, SHA_AMBIG2, SHA_BYNAME, SHA_BYNAME2):
        assert f"\n    {sha}" in out
    assert f"    {SHA_BYNAME}  {BYNAME_REASON}" in out and f"    {SHA_BYNAME2}  {BYNAME_REASON_2}" in out
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["ambiguous_rows"] == 4
    assert {o["object_id"]: o["unresolved"] for a in j["ambiguous"] for o in a["objects"]} == \
        {SHA_AMBIG: None, SHA_AMBIG2: None, SHA_BYNAME: BYNAME_REASON, SHA_BYNAME2: BYNAME_REASON_2}
    # the apply and the verify leave every ambiguous row as it was: listed by name, never corrected
    before = {sha: _live(root, sha) for sha in (SHA_AMBIG, SHA_AMBIG2, SHA_BYNAME, SHA_BYNAME2)}
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert len(app.written) == 2 and [g.members for g in app.ambiguous] == [unstated_rows, byname_rows]
    for sha in before:
        assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{sha[:16]}"))
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert {sha: _live(root, sha) for sha in before} == before
    ver = RC.verify_fit_effective_batch(root)
    assert [g.members for g in ver.ambiguous] == [unstated_rows, byname_rows] and not any(sha in RC.format_batch_report(ver) for sha in before)
    _rebuild_equal(root, tmp_path)


def test_a_dose_copy_left_null_by_name_with_nothing_recorded_is_listed_by_its_entry_not_dropped(tmp_path, capsys):
    """Review round astra (2026-10-13). The adapter's NULL by name has three shapes whose folded block records NO effective_batch: the
    copied record's key absent (NOKEY: the block without the key), the key null (NULLKEY: the block carries the null) and a copy with no
    copied record at all (NOBLOCK: a subtraction's dose child, no block). Each stands in the table with the explicit NULL, the `unverified`
    entry and the reason -- and a selection on the recorded value alone dropped all three from the text and the JSON (zero ambiguous, zero
    nothing to compare), so an operator could not find them from the report. They are selected by the ENTRY: listed under the by-name
    statement by object id with the adapter's reason, the recorded value `-`, in the text and the JSON; never corrected (`--apply` writes
    nothing for them); the verify is clean (a NULL is no disagreement) and still names them in its object; none is a parent cross-check
    (no derived value). Outside the class, pinned beside them: UNSAID (a dose copy with NULL, no entry and nothing recorded: nothing to
    read) and ANCHORISH (another derivation's row left NULL by its own writer, the same `unverified` shape: not this pass's question)."""
    root = _corpus(tmp_path, with_unrecorded_by_name=True)
    for sha in (SHA_NOKEY, SHA_NULLKEY, SHA_NOBLOCK, SHA_UNSAID, SHA_ANCHORISH):
        assert _live(root, sha)["feb"] is None, sha
    assert _live(root, SHA_NOKEY)["hp"] == {k: v for k, v in HOOK_BLOCK.items() if k != "effective_batch"}
    assert _live(root, SHA_NULLKEY)["hp"] == {**HOOK_BLOCK, "effective_batch": None} and _live(root, SHA_NOBLOCK)["hp"] is None
    rep = RC.reconcile_fit_effective_batch(root)
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and rep.unclassified == [] and rep.parent_checks == []
    expected = {SHA_NOKEY: nokey_reason(synth.OBJECT_SHA), SHA_NULLKEY: nokey_reason(SHA_FINE), SHA_NOBLOCK: noblock_reason(SHA_FINE)}
    listed = {oid: why for g in rep.ambiguous for oid, why in g.members}
    assert listed == {SHA_AMBIG: None, **expected}
    assert SHA_UNSAID not in listed and SHA_ANCHORISH not in listed
    # the groups: AMBIG's (recorded 8, no entry) beside two by-name groups whose recorded value is None -- the subtraction child's writer, then the
    # psr writer with its two copies (the SQL's order: writer, then recorded, NULL first)
    assert [g.summary for g in rep.ambiguous] == [("scripts/sdfbase/lora_subtract.py", None, synth.FIT_MODEL_ID, "unverified", 1),
                                                   ("scripts/sdfbase/scale_lora_dose.py", 8, synth.FIT_MODEL_ID, None, 1),
                                                   ("scripts/sdfbase/scale_psr_dose.py", None, synth.FIT_MODEL_ID, "unverified", 2)]
    child, _ambig, psr = rep.ambiguous
    assert child.members == [(SHA_NOBLOCK, noblock_reason(SHA_FINE))] and child.statement == RC.BATCH_AMBIGUOUS_BY_NAME
    assert psr.members == sorted([(SHA_NOKEY, nokey_reason(synth.OBJECT_SHA)), (SHA_NULLKEY, nokey_reason(SHA_FINE))]) and psr.statement == RC.BATCH_AMBIGUOUS_BY_NAME
    assert rep.nothing_to_compare == 1                      # NONE, as before: the by-name rows are ambiguous, not nothing-to-compare
    d = rep.as_dict()
    assert d["ambiguous_rows"] == 4 and {o["object_id"]: o["unresolved"] for a in d["ambiguous"] for o in a["objects"]} == listed
    assert [a["recorded"] for a in d["ambiguous"]] == [None, 8, None]
    # the text: `-` where nothing is recorded, every row under its group with its reason
    text = RC.format_batch_report(rep)
    assert "or a dose copy's NULL stated by the adapter's own entry with none recorded" in text and "listed by object id, not corrected): 4 row(s)" in text
    child_line = f"  {'scripts/sdfbase/lora_subtract.py':<40} {'-':<4} {synth.FIT_MODEL_ID:<20} {1:>6}  {RC.BATCH_AMBIGUOUS_BY_NAME}"
    psr_line = f"  {'scripts/sdfbase/scale_psr_dose.py':<40} {'-':<4} {synth.FIT_MODEL_ID:<20} {2:>6}  {RC.BATCH_AMBIGUOUS_BY_NAME}"
    assert f"{child_line}\n    {SHA_NOBLOCK}  {noblock_reason(SHA_FINE)}\n" in text
    assert psr_line + "".join(f"\n    {oid}  {why}" for oid, why in psr.members) + "\nnothing to compare" in text
    assert SHA_UNSAID not in text and SHA_ANCHORISH not in text
    # the CLI's text and JSON carry the same names
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    for sha, why in expected.items():
        assert f"\n    {sha}  {why}" in out
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["ambiguous_rows"] == 4 and {o["object_id"]: o["unresolved"] for a in j["ambiguous"] for o in a["objects"]} == listed
    # the apply writes nothing for them; the verify is clean and still names them; nothing of the five rows moves
    before = {sha: _live(root, sha) for sha in (SHA_NOKEY, SHA_NULLKEY, SHA_NOBLOCK, SHA_UNSAID, SHA_ANCHORISH)}
    app = RC.reconcile_fit_effective_batch(root, apply=True)
    assert len(app.written) == 2 and app.held == [] and {oid for g in app.ambiguous for oid, _w in g.members} == set(listed)
    for sha in before:
        assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{sha[:16]}")), sha
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert {sha: _live(root, sha) for sha in before} == before
    ver = RC.verify_fit_effective_batch(root)
    assert ver.rows == [] and ver.listed == [] and ver.parent_checks == [] and {oid for g in ver.ambiguous for oid, _w in g.members} == set(listed)
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 0
    assert "inconsistent 0 row(s); the class is consistent" in capsys.readouterr().out
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the parent cross-check (the registry owner's single-source decision, 2026-10-13): read-only, listed by name, never written for
# --------------------------------------------------------------------------------------------


PARENT_HEAD = ("parent cross-check (read-only: every dose copy whose derived fit_effective_batch is read against its PARENT object row's recorded "
               "effective batch; a disagreement is listed by name and never written for)")


def test_a_derived_dose_copy_is_read_against_its_parent_row_and_a_disagreement_is_listed_by_name_never_written(tmp_path, capsys):
    """The adapter states a dose copy's batch from the ONE record copied beside the copy, so whether that copy is faithful to the parent the
    stamp names is the registry's to say from the two rows it holds. Every derived dose copy is read against its parent row: AGREE (64 =
    FINE's recorded 64) and VIACOL (32 = NONE's column, the parent recording no block value) agree; RULED's parent states no batch and
    ORPHAN's parent is no row of the registry -- neither a disagreement, each said so; PCHECK (derived 8 = its own block, in neither of the
    pass's classes) disagrees with FINE's 64, and DERIVED_ODD (derived 1, parent EPOCH 64) disagrees too, beside being unclassified. A
    disagreement is listed by object id with both values in the dry run and the verify (exit 3) and `--apply` writes nothing for it: the
    entry is recognised by its confidence, whichever writer stated it."""
    root = _corpus(tmp_path, with_ruled=True, with_parent_check=True, with_parent_disagreement=True)
    rep = RC.reconcile_fit_effective_batch(root)
    checks = {c.object_id: c for c in rep.parent_checks}
    assert set(checks) == {SHA_RULED, SHA_DERIVED_ODD, SHA_AGREE, SHA_VIACOL, SHA_PCHECK, SHA_ORPHAN} and [c.object_id for c in rep.parent_checks] == sorted(checks)
    agree, viacol, ruled, orphan, pcheck, odd = (checks[s] for s in (SHA_AGREE, SHA_VIACOL, SHA_RULED, SHA_ORPHAN, SHA_PCHECK, SHA_DERIVED_ODD))
    assert (agree.status, agree.derived, agree.parent_object_id, agree.parent_present, agree.parent_value, agree.parent_source) == \
        (RC.BATCH_PARENT_AGREES, 64, SHA_FINE, True, 64, "the parent's fit_hyperparameters_json.effective_batch")
    assert (viacol.status, viacol.derived, viacol.parent_value, viacol.parent_source) == (RC.BATCH_PARENT_AGREES, 32, 32, "the parent's fit_effective_batch")
    assert (ruled.status, ruled.parent_object_id, ruled.parent_present, ruled.parent_value, ruled.parent_source) == (RC.BATCH_PARENT_STATES_NONE, synth.OBJECT_SHA, True, None, None)
    assert (orphan.status, orphan.parent_object_id, orphan.parent_present, orphan.parent_value) == (RC.BATCH_PARENT_ABSENT, SHA_NOWHERE, False, None)
    assert (pcheck.status, pcheck.derived, pcheck.parent_object_id, pcheck.parent_value, pcheck.fit_writer) == \
        (RC.BATCH_PARENT_DISAGREES, 8, SHA_FINE, 64, "scripts/sdfbase/scale_lora_dose.py")
    assert odd.disagrees and (odd.derived, odd.parent_object_id, odd.parent_value) == (1, SHA_EPOCH, 64)
    assert not agree.disagrees and not viacol.disagrees and not ruled.disagrees and not orphan.disagrees
    assert [c.object_id for c in rep.parent_disagreements] == sorted([SHA_DERIVED_ODD, SHA_PCHECK]) and [c.object_id for c in rep.parent_absent] == [SHA_ORPHAN]
    assert rep.parent_counts() == {RC.BATCH_PARENT_AGREES: 2, RC.BATCH_PARENT_DISAGREES: 2, RC.BATCH_PARENT_NONINTEGRAL: 0, RC.BATCH_PARENT_ABSENT: 1,
                                   RC.BATCH_PARENT_STATES_NONE: 1, RC.BATCH_PARENT_UNNAMED: 0}
    # PCHECK is in neither of the pass's classes (its own two values agree) and is no correction candidate; the disagreement is the parent's
    assert SHA_PCHECK not in {r.object_id for r in rep.rows} and SHA_PCHECK not in {oid for g in rep.ambiguous for oid in g.object_ids}
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and [r.object_id for r in rep.unclassified] == [SHA_DERIVED_ODD]
    d = rep.as_dict()["parent_check"]
    assert d["copies"] == 6 and d["disagreements"] == 2 and d["by_status"] == rep.parent_counts()
    assert {c["object_id"]: c["status"] for c in d["copies_by_name"]} == {s: checks[s].status for s in checks}
    assert next(c for c in d["copies_by_name"] if c["object_id"] == SHA_PCHECK) == pcheck.as_dict() == \
        {"object_id": SHA_PCHECK, "derived": 8, "fit_writer": "scripts/sdfbase/scale_lora_dose.py", "parent_object_id": SHA_FINE, "parent_present": True,
         "parent_value": 64, "parent_source": "the parent's fit_hyperparameters_json.effective_batch", "status": RC.BATCH_PARENT_DISAGREES}
    # the dry run's text: the counts, then every disagreement by name with both values, then every absent parent by name; agreeing copies
    # and parents stating none are counted, not listed
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert (f"{PARENT_HEAD}: 6 derived dose copies -- agree 2, disagree 2, not an integer in its own type 0 (a disagreement: nothing is coerced), "
            "parent row absent from the registry 1 (not a disagreement), parent states no batch 1, no parent named 0") in out
    pcheck_line = f"  {SHA_PCHECK} derived 8 [scripts/sdfbase/scale_lora_dose.py] != parent {SHA_FINE[:16]} 64 (the parent's fit_hyperparameters_json.effective_batch)"
    odd_line = f"  {SHA_DERIVED_ODD} derived 1 [scripts/sdfbase/scale_relu_dose.py] != parent {SHA_EPOCH[:16]} 64 (the parent's fit_hyperparameters_json.effective_batch)"
    orphan_line = f"  {SHA_ORPHAN} derived 8 [scripts/sdfbase/scale_lora_dose.py]: parent {SHA_NOWHERE[:16]} absent from the registry (not a disagreement: nothing to compare)"
    assert f"{odd_line}\n{pcheck_line}\n{orphan_line}\n" in out and pcheck.line == pcheck_line.strip() and orphan.line == orphan_line.strip()
    tail = out.split(PARENT_HEAD)[1]
    assert SHA_AGREE not in tail and SHA_VIACOL not in tail and SHA_RULED not in tail
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)["parent_check"]
    assert j["copies"] == 6 and j["disagreements"] == 2 and {c["object_id"] for c in j["copies_by_name"] if c["status"] == RC.BATCH_PARENT_DISAGREES} == {SHA_PCHECK, SHA_DERIVED_ODD}
    # the apply: the two hook rows written; nothing for a parent disagreement, an absent parent or an agreeing copy (exit 2 is DERIVED_ODD's refusal)
    before = {sha: _live(root, sha) for sha in (SHA_PCHECK, SHA_ORPHAN, SHA_AGREE, SHA_VIACOL, SHA_RULED)}
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 2
    out = capsys.readouterr().out
    assert f"{odd_line}\n{pcheck_line}\n{orphan_line}\n" in out
    app = RC.reconcile_fit_effective_batch(root, apply=False)
    assert [c.object_id for c in app.parent_disagreements] == sorted([SHA_DERIVED_ODD, SHA_PCHECK])
    for sha in before:
        assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{sha[:16]}")), sha
    assert sorted(os.listdir(os.path.join(root, "records", "registry_owner"))) == sorted([f"object__{SHA_EPOCH[:16]}", f"object__{SHA_DOSE[:16]}"])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert {sha: _live(root, sha) for sha in before} == before and _live(root, SHA_EPOCH)["feb"] == 64
    # the verify: the hook rows are consistent now; the parent disagreements stand (exit 3), PCHECK by name beside the unclassified
    # DERIVED_ODD -- one row each -- and ORPHAN said absent; AGREE, VIACOL and RULED are not listed
    ver = RC.verify_fit_effective_batch(root)
    assert [r.object_id for r in ver.rows] == [SHA_DERIVED_ODD] and [c.object_id for c in ver.parent_disagreements] == sorted([SHA_DERIVED_ODD, SHA_PCHECK])
    assert [getattr(x, "object_id") for x in ver.listed] == [SHA_DERIVED_ODD, SHA_PCHECK]        # an object once, whichever ways it disagrees
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"{odd_line}\n{pcheck_line}\n{orphan_line}\ninconsistent 2 row(s)" in out and "derived dose copies whose fit_effective_batch disagrees with their PARENT row" in out
    assert SHA_AGREE not in out and SHA_VIACOL not in out and SHA_RULED not in out
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify", "--json"]) == 3
    j = json.loads(capsys.readouterr().out)
    assert j["verify"] and j["parent_check"]["disagreements"] == 2
    _rebuild_equal(root, tmp_path)


def test_a_parent_row_absent_from_the_registry_is_not_a_disagreement_and_the_verify_is_clean(tmp_path, capsys):
    """ORPHAN's parent is no row of the registry: nothing to compare, said so by name in the dry run and the verify, never a failure --
    with the hook rows corrected and folded, the verify exits 0 and reads the class consistent while the absent parent is still named;
    AGREE and VIACOL (agreeing through the parent's block, then through its column) are counted and not listed. Nothing is written for any
    of the three, on `--apply` or at all."""
    root = _corpus(tmp_path, with_parent_check=True)
    rep = RC.reconcile_fit_effective_batch(root)
    assert [c.object_id for c in rep.parent_checks] == sorted([SHA_AGREE, SHA_VIACOL, SHA_ORPHAN]) and rep.parent_disagreements == []
    assert [c.object_id for c in rep.parent_absent] == [SHA_ORPHAN]
    assert rep.parent_counts() == {RC.BATCH_PARENT_AGREES: 2, RC.BATCH_PARENT_DISAGREES: 0, RC.BATCH_PARENT_NONINTEGRAL: 0, RC.BATCH_PARENT_ABSENT: 1,
                                   RC.BATCH_PARENT_STATES_NONE: 0, RC.BATCH_PARENT_UNNAMED: 0}
    orphan_line = f"  {SHA_ORPHAN} derived 8 [scripts/sdfbase/scale_lora_dose.py]: parent {SHA_NOWHERE[:16]} absent from the registry (not a disagreement: nothing to compare)"
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert (f"{PARENT_HEAD}: 3 derived dose copies -- agree 2, disagree 0, not an integer in its own type 0 (a disagreement: nothing is coerced), "
            f"parent row absent from the registry 1 (not a disagreement), parent states no batch 0, no parent named 0\n{orphan_line}\n") in out
    before = {sha: _live(root, sha) for sha in (SHA_AGREE, SHA_VIACOL, SHA_ORPHAN)}
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 0
    capsys.readouterr()
    assert sorted(os.listdir(os.path.join(root, "records", "registry_owner"))) == sorted([f"object__{SHA_EPOCH[:16]}", f"object__{SHA_DOSE[:16]}"])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert {sha: _live(root, sha) for sha in before} == before
    ver = RC.verify_fit_effective_batch(root)
    assert ver.rows == [] and ver.listed == [] and [c.object_id for c in ver.parent_absent] == [SHA_ORPHAN]
    text = RC.format_batch_report(ver)
    assert text.endswith(f"{orphan_line}\ninconsistent 0 row(s); the class is consistent") and SHA_AGREE not in text and SHA_VIACOL not in text
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 0
    assert orphan_line in capsys.readouterr().out
    # the verify also reads a copy that names no parent as nothing to compare (the shape cannot fold from a dose stamp, so the predicate
    # alone is pinned here)
    assert RC.BatchParentCheck("x" * 64, 8, None, None, False).status == RC.BATCH_PARENT_UNNAMED and not RC.BatchParentCheck("x" * 64, 8, None, None, False).disagrees
    numeric_string = RC.BatchParentCheck("x" * 64, 8, None, "y" * 64, True, "8", "the parent's fit_effective_batch")
    assert numeric_string.disagrees and numeric_string.status == RC.BATCH_PARENT_NONINTEGRAL       # a numeric string is not 8: nothing coerced
    _rebuild_equal(root, tmp_path)


def test_a_parent_recording_a_non_integer_effective_batch_is_a_disagreement_of_type_never_read_as_agreement(tmp_path, capsys):
    """Review round sol (2026-10-13). CFLOAT derives 8 from its copied record while its parent PFLOAT's block records 8.0; CBOOL derives
    1 while its parent PBOOL's block records true. Python's `8.0 == 8` and `True == 1` hold, so a comparison by `==` alone read both as
    agreement: the dry run named no discrepancy and `--verify` exited 0 past a parent record whose batch had changed type -- the very
    values the adapter refuses by name when it reads the copied record. The cross-check reads the two values in their own type: a parent
    value that is not an integer (a float, a boolean, a numeric string) is `BATCH_PARENT_NONINTEGRAL`, a disagreement -- listed by object
    id with the value and its type in the dry run and the verify (exit 3), counted beside the other disagreements, never written for.
    The parents themselves stand in neither of the pass's classes (its class SQL reads 8 and 8.0 as equal), so the cross-check is the one
    place the type shows."""
    root = _corpus(tmp_path, with_parent_check=True, with_parent_nonintegral=True)
    for sha, feb, rec in ((SHA_PFLOAT, 8, 8.0), (SHA_PBOOL, 1, True)):
        live = _live(root, sha)
        assert live["feb"] == feb and live["hp"]["effective_batch"] == rec and type(live["hp"]["effective_batch"]) is type(rec), sha
    rep = RC.reconcile_fit_effective_batch(root)
    checks = {c.object_id: c for c in rep.parent_checks}
    assert set(checks) == {SHA_AGREE, SHA_VIACOL, SHA_ORPHAN, SHA_CFLOAT, SHA_CBOOL}
    cfloat, cbool = checks[SHA_CFLOAT], checks[SHA_CBOOL]
    assert (cfloat.status, cfloat.derived, cfloat.parent_object_id, cfloat.parent_present, cfloat.parent_value, cfloat.parent_source) == \
        (RC.BATCH_PARENT_NONINTEGRAL, 8, SHA_PFLOAT, True, 8.0, "the parent's fit_hyperparameters_json.effective_batch")
    assert type(cfloat.parent_value) is float and cfloat.disagrees
    assert (cbool.status, cbool.derived, cbool.parent_object_id, cbool.parent_value) == (RC.BATCH_PARENT_NONINTEGRAL, 1, SHA_PBOOL, True)
    assert type(cbool.parent_value) is bool and cbool.disagrees
    assert [c.object_id for c in rep.parent_disagreements] == sorted([SHA_CFLOAT, SHA_CBOOL]) and [c.object_id for c in rep.parent_absent] == [SHA_ORPHAN]
    assert rep.parent_counts() == {RC.BATCH_PARENT_AGREES: 2, RC.BATCH_PARENT_DISAGREES: 0, RC.BATCH_PARENT_NONINTEGRAL: 2, RC.BATCH_PARENT_ABSENT: 1,
                                   RC.BATCH_PARENT_STATES_NONE: 0, RC.BATCH_PARENT_UNNAMED: 0}
    # the parents and the copies are in neither of the pass's own classes: the class SQL reads 8 and 8.0 (1 and true) as equal
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and rep.unclassified == []
    assert not ({SHA_PFLOAT, SHA_PBOOL, SHA_CFLOAT, SHA_CBOOL} & ({r.object_id for r in rep.rows} | {oid for g in rep.ambiguous for oid in g.object_ids}))
    d = rep.as_dict()["parent_check"]
    assert d["copies"] == 5 and d["disagreements"] == 2 and d["by_status"] == rep.parent_counts()
    assert {c["object_id"] for c in d["copies_by_name"] if c["status"] == RC.BATCH_PARENT_NONINTEGRAL} == {SHA_CFLOAT, SHA_CBOOL}
    assert next(c for c in d["copies_by_name"] if c["object_id"] == SHA_CFLOAT)["parent_value"] == 8.0
    # the text: counted beside the disagreements, each named with the value and its type, before the absent parent
    cfloat_line = (f"  {SHA_CFLOAT} derived 8 [scripts/sdfbase/scale_lora_dose.py] vs parent {SHA_PFLOAT[:16]} 8.0 (the parent's "
                   "fit_hyperparameters_json.effective_batch): the parent's 8.0 (float) not an integer in its own type; nothing is coerced, so this is not an agreement")
    cbool_line = (f"  {SHA_CBOOL} derived 1 [scripts/sdfbase/scale_lora_dose.py] vs parent {SHA_PBOOL[:16]} True (the parent's "
                  "fit_hyperparameters_json.effective_batch): the parent's True (bool) not an integer in its own type; nothing is coerced, so this is not an agreement")
    orphan_line = f"  {SHA_ORPHAN} derived 8 [scripts/sdfbase/scale_lora_dose.py]: parent {SHA_NOWHERE[:16]} absent from the registry (not a disagreement: nothing to compare)"
    assert cfloat.line == cfloat_line.strip() and cbool.line == cbool_line.strip()
    listed = "\n".join(line for _sha, line in sorted([(SHA_CFLOAT, cfloat_line), (SHA_CBOOL, cbool_line)]))     # object-id order, as the report lists them
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert (f"{PARENT_HEAD}: 5 derived dose copies -- agree 2, disagree 0, not an integer in its own type 2 (a disagreement: nothing is coerced), "
            f"parent row absent from the registry 1 (not a disagreement), parent states no batch 0, no parent named 0\n{listed}\n{orphan_line}\n") in out
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)["parent_check"]
    assert j["disagreements"] == 2 and j["by_status"][RC.BATCH_PARENT_NONINTEGRAL] == 2
    assert {c["object_id"]: c["parent_value"] for c in j["copies_by_name"] if c["status"] == RC.BATCH_PARENT_NONINTEGRAL} == {SHA_CFLOAT: 8.0, SHA_CBOOL: True}
    # the apply writes the two hook rows and nothing for the parents or their copies; the verify lists both copies by name (exit 3)
    before = {sha: _live(root, sha) for sha in (SHA_PFLOAT, SHA_PBOOL, SHA_CFLOAT, SHA_CBOOL)}
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 0
    assert f"{listed}\n{orphan_line}\n" in capsys.readouterr().out
    assert sorted(os.listdir(os.path.join(root, "records", "registry_owner"))) == sorted([f"object__{SHA_EPOCH[:16]}", f"object__{SHA_DOSE[:16]}"])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert {sha: _live(root, sha) for sha in before} == before
    ver = RC.verify_fit_effective_batch(root)
    assert ver.rows == [] and [c.object_id for c in ver.parent_disagreements] == sorted([SHA_CFLOAT, SHA_CBOOL])
    assert [getattr(x, "object_id") for x in ver.listed] == sorted([SHA_CFLOAT, SHA_CBOOL])
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"{listed}\n{orphan_line}\ninconsistent 2 row(s)" in out and "(a different integer, or a value not an integer in its own type: nothing is coerced)" in out
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify", "--json"]) == 3
    j = json.loads(capsys.readouterr().out)
    assert j["verify"] and j["parent_check"]["disagreements"] == 2 and j["parent_check"]["by_status"][RC.BATCH_PARENT_NONINTEGRAL] == 2
    # the predicate alone, both sides: an integer pair agrees or disagrees by value; a float, a boolean or a numeric string on either side is a
    # disagreement of type, and `8.0 == 8` / `True == 1` never make one an agreement
    def status(derived, parent):
        return RC.BatchParentCheck("x" * 64, derived, None, "y" * 64, True, parent, "the parent's fit_effective_batch").status
    assert status(8, 8) == RC.BATCH_PARENT_AGREES and status(8, 16) == RC.BATCH_PARENT_DISAGREES
    assert status(8, 8.0) == status(1, True) == status(8, "8") == status(8, 8.5) == status(8.0, 8) == status(True, 1) == RC.BATCH_PARENT_NONINTEGRAL
    assert all(RC.BatchParentCheck("x" * 64, dv, None, "y" * 64, True, pv, "the parent's fit_effective_batch").disagrees for dv, pv in ((8, 8.0), (1, True), (8, "8")))
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the statement check (review round sol-2, 2026-10-13): the adapter's statement against the copy's OWN row, in Python, in its own type
# --------------------------------------------------------------------------------------------


STATEMENT_HEAD = ("statement check (read-only: every dose copy whose fit_effective_batch carries the adapter's derived entry, or its NULL-by-name entry "
                  "with a value standing beside it, read against the row's OWN column and block in their own type; an inconsistent copy is listed by name "
                  "and never written for)")


def _statement_line(sha, stated, column, writer, block, status):
    return f"{sha} {stated} entry, column {column} [{writer}], block effective_batch {block}: {status}; nothing is coerced, never written for"


def test_a_derived_copy_inconsistent_with_its_own_block_is_listed_by_name_and_fails_the_verify_even_where_sql_reads_the_pair_equal(tmp_path, capsys):
    """Review round sol-2 (2026-10-13). The class SQL selects a row whose two values DISAGREE by SQLite's equality, which reads 8 and 8.0 as
    one value and a missing key as nothing to compare -- so a dose copy claiming derivation from a record the adapter itself would refuse
    by name was never selected, `batch_derived_reason` never saw it, its parent could still read `agrees` (or state nothing), and --verify
    exited 0 past it. Four such copies, none selected by the class SQL and none a parent disagreement: SFLOAT (derived 8, the block moved to
    8.0 by another object writer), SNOKEY (derived 8, the block without the key -- counted as nothing to compare before, now named instead),
    SNULL (a `derived` entry beside a NULL column, the block recording 8: in the ambiguous class by its entry, and inconsistent here) and
    SSTALE (the adapter's `unverified` NULL-by-name entry beside a standing 8 whose block records 8: the two values agree, so the class SQL
    never saw the retained value). The statement check reads every statement-carrying dose copy against its own column and block in Python,
    in their own type: each of the four is listed by object id with which side fails, in the dry run and the verify (exit 3), nothing is
    written for them on --apply, and the copies the class SQL does select (DERIVED_ODD, STALE) are one row each in the verify's count, while
    RULED (derived 8 = the block's 8) is counted consistent and not listed."""
    root = _corpus(tmp_path, with_ruled=True, with_stale=True, with_statement_check=True)
    assert _live(root, SHA_SFLOAT)["feb"] == 8 and _live(root, SHA_SFLOAT)["hp"]["effective_batch"] == 8.0 and type(_live(root, SHA_SFLOAT)["hp"]["effective_batch"]) is float
    assert _live(root, SHA_SNOKEY)["feb"] == 8 and "effective_batch" not in _live(root, SHA_SNOKEY)["hp"]
    assert _live(root, SHA_SNULL)["feb"] is None and _live(root, SHA_SNULL)["hp"]["effective_batch"] == 8 and _live(root, SHA_SNULL)["conf"]["fit_effective_batch"] == DERIVED_ENTRY
    assert _live(root, SHA_SSTALE)["feb"] == 8 and _live(root, SHA_SSTALE)["hp"]["effective_batch"] == 8 and _live(root, SHA_SSTALE)["conf"]["fit_effective_batch"] == BYNAME_ENTRY_2
    rep = RC.reconcile_fit_effective_batch(root)
    # the class SQL selects none of the four: the clearly-wrong rows and the two refused disagreements are as before
    assert [r.object_id for r in rep.selected] == sorted([SHA_EPOCH, SHA_DOSE]) and [r.object_id for r in rep.unclassified] == sorted([SHA_DERIVED_ODD, SHA_STALE])
    assert not ({SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE} & {r.object_id for r in rep.rows})
    # the statement check: every statement-carrying copy, in object-id order; one consistent (RULED), six inconsistent, each named with its side
    checks = {c.object_id: c for c in rep.statement_checks}
    assert set(checks) == {SHA_RULED, SHA_DERIVED_ODD, SHA_STALE, SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE} and [c.object_id for c in rep.statement_checks] == sorted(checks)
    assert SHA_BYNAME not in checks            # the adapter's unverified entry beside NULL: its statement to the letter, in the ambiguous class
    assert (checks[SHA_RULED].status, checks[SHA_RULED].value, checks[SHA_RULED].recorded, checks[SHA_RULED].inconsistent) == (RC.BATCH_STATEMENT_CONSISTENT, 8, 8, False)
    sfloat, snokey, snull, sstale = (checks[s] for s in (SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE))
    assert (sfloat.status, sfloat.stated, sfloat.value, sfloat.recorded, sfloat.block_present) == (RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL, "derived", 8, 8.0, True)
    assert type(sfloat.recorded) is float and sfloat.inconsistent and sfloat.block_text == "8.0 (float)"
    assert (snokey.status, snokey.value, snokey.recorded, snokey.block_present, snokey.block_text) == (RC.BATCH_STATEMENT_BLOCK_NONE, 8, None, True, "-")
    assert (snull.status, snull.value, snull.recorded) == (RC.BATCH_STATEMENT_DERIVED_NULL, None, 8)
    assert (sstale.status, sstale.stated, sstale.value, sstale.recorded) == (RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME, "unverified", 8, 8)
    assert checks[SHA_DERIVED_ODD].status == RC.BATCH_STATEMENT_BLOCK_DISAGREES and checks[SHA_STALE].status == RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME
    assert [c.object_id for c in rep.statement_inconsistencies] == sorted([SHA_DERIVED_ODD, SHA_STALE, SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE])
    assert rep.statement_counts() == {RC.BATCH_STATEMENT_CONSISTENT: 1, RC.BATCH_STATEMENT_DERIVED_NULL: 1, RC.BATCH_STATEMENT_DERIVED_NONINTEGRAL: 0,
                                      RC.BATCH_STATEMENT_BLOCK_NONE: 1, RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL: 1, RC.BATCH_STATEMENT_BLOCK_DISAGREES: 1,
                                      RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME: 2}
    # the parent cross-check reads none of the four as a disagreement (SFLOAT's and SNOKEY's parent states no batch; SNULL and SSTALE have no
    # derived value), so the statement check is the one thing that names them
    assert {c.object_id: c.status for c in rep.parent_checks} == {SHA_RULED: RC.BATCH_PARENT_STATES_NONE, SHA_DERIVED_ODD: RC.BATCH_PARENT_DISAGREES,
                                                                   SHA_SFLOAT: RC.BATCH_PARENT_STATES_NONE, SHA_SNOKEY: RC.BATCH_PARENT_STATES_NONE}
    assert [c.object_id for c in rep.parent_disagreements] == [SHA_DERIVED_ODD]
    # SNULL is in the ambiguous class too (a dose copy's NULL beside an entry), under its own statement; SNOKEY is NOT counted as nothing to compare
    assert [g.summary for g in rep.ambiguous] == [("scripts/sdfbase/scale_lora_dose.py", 8, synth.FIT_MODEL_ID, None, 1),
                                                   ("scripts/sdfbase/scale_psr_dose.py", 8, synth.FIT_MODEL_ID, "derived", 1),
                                                   ("scripts/sdfbase/scale_psr_dose.py", 64.0, synth.FIT_MODEL_ID, "unverified", 1)]
    assert rep.ambiguous[1].members == [(SHA_SNULL, DERIVED_ENTRY["source"])] and rep.ambiguous[1].statement == "confidence entry 'derived' beside a NULL column (read the row)"
    assert rep.nothing_to_compare == 1                      # NONE alone: a derived dose copy with no block value is the statement check's, by name
    # the verify's list: every row once -- the four class rows, then the four statement inconsistencies the class did not name; the parent
    # disagreement (DERIVED_ODD) is already named
    assert [getattr(x, "object_id") for x in rep.listed] == sorted([SHA_EPOCH, SHA_DOSE, SHA_DERIVED_ODD, SHA_STALE]) + sorted([SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE])
    d = rep.as_dict()
    assert d["statement_check"]["copies"] == 7 and d["statement_check"]["inconsistent"] == 6 and d["statement_check"]["by_status"] == rep.statement_counts()
    assert {c["object_id"]: c["status"] for c in d["statement_check"]["copies_by_name"]} == {s: checks[s].status for s in checks}
    assert next(c for c in d["statement_check"]["copies_by_name"] if c["object_id"] == SHA_SFLOAT) == sfloat.as_dict() == \
        {"object_id": SHA_SFLOAT, "stated": "derived", "value": 8, "fit_writer": "scripts/sdfbase/scale_lora_dose.py", "block_present": True, "recorded": 8.0,
         "status": RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL}
    assert d["nothing_to_compare"] == 1
    # the text: the counts, then every inconsistent copy by name with its column and its block's value (each with its type) and which side
    # fails, after the nothing-to-compare line and before the parent cross-check; RULED is counted, not listed
    sfloat_line = _statement_line(SHA_SFLOAT, "derived", "8 (int)", "scripts/sdfbase/scale_lora_dose.py", "8.0 (float)", RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL)
    snokey_line = _statement_line(SHA_SNOKEY, "derived", "8 (int)", "scripts/sdfbase/scale_lora_dose.py", "-", RC.BATCH_STATEMENT_BLOCK_NONE)
    snull_line = _statement_line(SHA_SNULL, "derived", "NULL", "scripts/sdfbase/scale_psr_dose.py", "8 (int)", RC.BATCH_STATEMENT_DERIVED_NULL)
    sstale_line = _statement_line(SHA_SSTALE, "unverified", "8 (int)", "scripts/sdfbase/scale_lora_dose.py", "8 (int)", RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME)
    odd_line = _statement_line(SHA_DERIVED_ODD, "derived", "1 (int)", "scripts/sdfbase/scale_relu_dose.py", "64 (int)", RC.BATCH_STATEMENT_BLOCK_DISAGREES)
    stale_line = _statement_line(SHA_STALE, "unverified", "8 (int)", "scripts/sdfbase/scale_lora_dose.py", "64 (int)", RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME)
    by_line = {SHA_SFLOAT: sfloat_line, SHA_SNOKEY: snokey_line, SHA_SNULL: snull_line, SHA_SSTALE: sstale_line, SHA_DERIVED_ODD: odd_line, SHA_STALE: stale_line}
    assert sfloat.line == sfloat_line and snokey.line == snokey_line and snull.line == snull_line and sstale.line == sstale_line
    head = (f"{STATEMENT_HEAD}: 7 dose copies -- consistent 1, {RC.BATCH_STATEMENT_DERIVED_NULL} 1, {RC.BATCH_STATEMENT_DERIVED_NONINTEGRAL} 0, "
            f"{RC.BATCH_STATEMENT_BLOCK_NONE} 1, {RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL} 1, {RC.BATCH_STATEMENT_BLOCK_DISAGREES} 1, "
            f"{RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME} 2 (nothing is coerced: 8 is not 8.0)")
    listed = "".join(f"\n  {by_line[sha]}" for sha in sorted(by_line))
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch"]) == 0
    out = capsys.readouterr().out
    assert f"{head}{listed}\n{PARENT_HEAD}" in out and out.index("nothing to compare") < out.index(STATEMENT_HEAD) < out.index(PARENT_HEAD)
    assert "nothing to compare (fit_effective_batch set, no recorded effective_batch): 1 row(s)" in out
    assert SHA_RULED not in out.split(STATEMENT_HEAD)[1].split(PARENT_HEAD)[0]
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--dry-run", "--json"]) == 0
    j = json.loads(capsys.readouterr().out)
    assert j["statement_check"]["copies"] == 7 and j["statement_check"]["inconsistent"] == 6 and j["nothing_to_compare"] == 1
    assert {c["object_id"] for c in j["statement_check"]["copies_by_name"] if c["status"] != RC.BATCH_STATEMENT_CONSISTENT} == set(by_line)
    # the apply: the two hook rows written, the two class disagreements refused (exit 2); nothing for the four, whose rows do not move
    before = {sha: _live(root, sha) for sha in (SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE, SHA_RULED)}
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--apply"]) == 2
    assert f"{head}{listed}\n" in capsys.readouterr().out
    app = RC.reconcile_fit_effective_batch(root, apply=False)
    assert [c.object_id for c in app.statement_inconsistencies] == sorted(by_line)
    for sha in before:
        assert not os.path.isdir(os.path.join(root, "records", "registry_owner", f"object__{sha[:16]}")), sha
    assert sorted(os.listdir(os.path.join(root, "records", "registry_owner"))) == sorted([f"object__{SHA_EPOCH[:16]}", f"object__{SHA_DOSE[:16]}"])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.refused == 0 and rep2.blocked == 0, rep2.refusals
    assert {sha: _live(root, sha) for sha in before} == before and _live(root, SHA_EPOCH)["feb"] == 64
    # the verify: the hook rows are consistent now; the two refused class rows and the four statement inconsistencies stand, six rows once
    # each (exit 3), every one by name; RULED and BYNAME are not listed
    ver = RC.verify_fit_effective_batch(root)
    assert [r.object_id for r in ver.rows] == sorted([SHA_DERIVED_ODD, SHA_STALE])
    assert [getattr(x, "object_id") for x in ver.listed] == sorted([SHA_DERIVED_ODD, SHA_STALE]) + sorted([SHA_SFLOAT, SHA_SNOKEY, SHA_SNULL, SHA_SSTALE])
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"{head}{listed}\n" in out and "inconsistent 6 row(s)" in out
    assert "dose copies whose adapter statement does not hold of their own row" in out and SHA_RULED not in out and SHA_BYNAME not in out
    assert cli.main(["--root", root, "reconcile", "fit-effective-batch", "--verify", "--json"]) == 3
    j = json.loads(capsys.readouterr().out)
    assert j["verify"] and j["statement_check"]["inconsistent"] == 6 and j["statement_check"]["by_status"][RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL] == 1
    # the predicate alone: a derived column and a block value are consistent only as one integer (not a boolean) on both sides; every other
    # shape names its side, and `8.0 == 8` / `True == 1` never make one consistent; the adapter's NULL beside its unverified entry is consistent
    def status(stated, value, block_present, recorded):
        return RC.BatchStatementCheck("x" * 64, stated, value, None, block_present, recorded).status
    assert status("derived", 8, True, 8) == RC.BATCH_STATEMENT_CONSISTENT and status("derived", 64, True, 64) == RC.BATCH_STATEMENT_CONSISTENT
    assert status("derived", 8, True, 8.0) == status("derived", 1, True, True) == status("derived", 8, True, "8") == RC.BATCH_STATEMENT_BLOCK_NONINTEGRAL
    assert status("derived", 8, True, None) == status("derived", 8, False, None) == RC.BATCH_STATEMENT_BLOCK_NONE
    assert status("derived", 8, True, 16) == RC.BATCH_STATEMENT_BLOCK_DISAGREES
    assert status("derived", None, True, 8) == status("derived", None, False, None) == RC.BATCH_STATEMENT_DERIVED_NULL
    assert status("derived", 8.0, True, 8) == status("derived", 8.5, True, 8) == status("derived", True, True, 1) == RC.BATCH_STATEMENT_DERIVED_NONINTEGRAL
    assert status("unverified", 8, True, 8) == status("unverified", 8, True, 64) == status("unverified", 8, False, None) == RC.BATCH_STATEMENT_VALUE_BESIDE_NULL_BY_NAME
    assert status("unverified", None, True, 8) == RC.BATCH_STATEMENT_CONSISTENT
    assert RC.BatchStatementCheck("x" * 64, "derived", 8, None, False, None).block_text == "no block"
    assert all(RC.BatchStatementCheck("x" * 64, "derived", 8, None, True, rec).inconsistent for rec in (8.0, True, "8", None, 16))
    assert not RC.BatchStatementCheck("x" * 64, "derived", 8, None, True, 8).inconsistent
    _rebuild_equal(root, tmp_path)
