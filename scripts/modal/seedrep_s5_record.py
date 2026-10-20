#!/usr/bin/env python3
"""RS/smokes.json's S5 block -- the S5 smoke's two MEASURED records, VALIDATED before they are written (never a bare scalar, never a
percentile, never a hand-mirrored value):

  reference  one per planned window class: `S5_REFERENCE_KEYS` exactly (`aggregate_decode_tok_s`, >= 20 min, a COMPLETE store,
             sampling coverage >= 0.8, the eligible interval count, W / wave / attempt held to the standup) -- `S5.references[]`
  window     the multi-turn window measurements (`seedrep_s5_window_measurement/1`) -- `S5.multiturn[]`.
             DERIVED, never ingested -- `--rows <the S5 probe standup ids>` resolves each row's ACCEPTED attempt off the LIVE accepted Dict,
             reads the PRODUCER's `audits/s5_window/<leg>/<cell>__s<s>.json` (written in-container by `seedrep_s5_measure.py`, bound under
             `modal_evidence.json[producer_evidence]`) through the Volume API, requires the parent arm AND both probe cells per leg with
             ZERO clamped calls on a COMPLETE store, and writes those records + the derivation (`S5.window_derivation`). `--record` is REFUSED
             on this subcommand by name: an operator-authored window record is not a measurement.
  bars       the utilisation verdict (`bars_met` from the utilisation table over the smoke, EVERY bar with its value, threshold
             and verdict, the table's path + sha256, the attempts read) -- `S5.bars` (`--go` refuses unless it is true)

Two wave identities: every record's `wave_uuid` is the smoke's EXECUTION wave (`RS/smokes/S5/wave.json`, the wave of the
standup that MEASURED); the block's WRAPPER binds those records to the PRODUCTION target (`production_wave_uuid` =
the plan's wave, `launch_plan_sha`, W). A record stamped with the production wave is refused: no smoke runs there.

    seedrep_s5_record.py reference --rs RS --plan RS/launch_plan_draft.json --record measured.json [--replace]
    seedrep_s5_record.py window    --rs RS --plan RS/launch_plan_draft.json --rows <the 4 S5 probe standup ids> [--replace]
                                   (DERIVED from the LIVE accepted Dict + the producer's bound audits/s5_window/ records, via the Volume API)
    seedrep_s5_record.py bars      --rs RS --plan RS/launch_plan_draft.json --record verdict.json
    seedrep_s5_record.py pass      --rs RS --plan RS/launch_plan_draft.json --smoke-id S1 --rows <standup ids>
                                   (DERIVED from the LIVE accepted Dict + the producer's files under the accepted attempt root, via the Volume API; write-once)
    seedrep_s5_record.py pass      --rs RS --plan RS/launch_plan_draft.json --smoke-id S1 --rows <standup ids> --prior RS/prior/<W7>[,RS/prior/<W7>...] [--tree <W checkout>]
                                   (the rows' accepted attempts resolved through the PRIOR trees' smoke Dicts, re-audited by THIS tree -- `lp.derive_smoke_pass_from_priors`)
    seedrep_s5_record.py w-gate    --rs RS --plan RS/launch_plan.json --baseline-sha <sha>   (DERIVED from /vol/cpu-suite/<W|baseline>/summary.json via the Volume API; write-once)
    seedrep_s5_record.py decode-proof --rs RS --tree <W checkout>                                                          (the eight rows by exact ref)
    seedrep_s5_record.py certify   --rs RS --plan RS/launch_plan.json --draft RS/launch_plan_draft.json --tree <W checkout>  -> RS/launch_certificate.json

THE LAUNCH CERTIFICATE: `certify` assembles S0 (RS/s0_gates.json), the S1-S4 PASS records, the S5 block, the W gate and the
decode-ref proof into ONE sealed, write-once record bound to W + the production wave + `launch_plan_sha` (and the draft's, with draft == final
re-checked); `--go` / `--legs` / `--reattempt` consume the certificate and NOTHING else, re-validating every component on its own content.

The record is the S5 smoke's OWN measurement (its engine counters over its complete store), written as JSON by the operator step; this
tool refuses what the utilisation audit would refuse and merges it atomically (tmp + rename). Exit 0 written, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seedrep_launch_plan as lp  # noqa: E402

SMOKES_JSON = "smokes.json"


def load_smokes(rs: Path) -> dict:
    p = Path(rs) / SMOKES_JSON
    if not p.is_file():
        return {}
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: not an object")
    return doc


def write_smokes(rs: Path, doc: dict) -> Path:
    p = Path(rs) / SMOKES_JSON
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")                  # the records keep their schema order (S5_REFERENCE_KEYS)
    os.replace(tmp, p)
    return p


S5_SMOKE_ID = "S5"


def execution_wave_for(rs: Path, smoke_id: str) -> str:
    """The smoke's EXECUTION wave: `RS/smokes/<id>/wave.json`, minted by the app when the smoke row was spawned. A smoke
    that never ran has no wave, and nothing is recorded for it -- the production wave is never a substitute."""
    wave = lp.smoke_execution_wave(rs, smoke_id)
    if wave is None:
        raise ValueError(f"no {lp.smoke_wave_path(rs, smoke_id)}: smoke {smoke_id!r} has not run (its execution wave is minted at --smoke); "
                         "an S5 record binds to the wave that MEASURED, never to the production wave")
    return wave


def record_s5(kind: str, *, rs: Path, plan: dict, record: dict, replace: bool = False, smoke_id: str = S5_SMOKE_ID, read_text=None) -> dict:
    """Validate + merge ONE record of ``kind`` (`reference` | `window` | `bars`) into RS/smokes.json; raises ValueError with the refusal
    named. Every record is held to the `S5` smoke's EXECUTION wave (its `wave_uuid` = the standup that measured), and the
    block's WRAPPER is bound to the PRODUCTION target (`production_wave_uuid` = the plan's wave, `launch_plan_sha`, W)."""
    doc = load_smokes(rs)
    execution_wave = execution_wave_for(rs, smoke_id)
    lp.s5_block_bind(doc, W=plan["W"], execution_wave_uuid=execution_wave, production_wave_uuid=plan["wave_uuid"], launch_plan_sha=plan["launch_plan_sha"])
    if kind == "reference":
        # a reference measured under a PRIOR tree / wave is admitted through its standup's records on the volume, held to this plan; the
        # Volume reader is opened on the FIRST such read only (a reference of this tree and wave never touches the volume)
        lp.s5_smokes_add_reference(doc, record, W=plan["W"], wave_uuid=execution_wave, replace=replace, plan=plan, read_text=read_text or _lazy_volume_reader(rs))
    elif kind == "window":
        lp.s5_smokes_add_window(doc, record, wave_uuid=execution_wave, replace=replace)
    elif kind == "bars":
        problems = lp.utilisation_verdict_problems(record, W=plan["W"], execution_wave_uuid=execution_wave)
        if problems:
            raise ValueError(f"refusing to record the S5 utilisation verdict: {problems}")
        doc["S5"]["bars"] = dict(record)
    else:
        raise ValueError(f"unknown record kind {kind!r}")
    write_smokes(rs, doc)
    return doc


W_GATE_JSON = "w_gate.json"
DECODE_PROOF_JSON = "decode_ref_proof.json"
OPERATOR_RECORD_REFUSAL = ("--record is refused on `window` (W16s5, B2): an operator-authored window record is not a measurement; the D50 records "
                           "are DERIVED from the accepted S5 probe attempts' producer files (`audits/s5_window/...`, written in-container by "
                           "seedrep_s5_measure.py) -- pass --rows <the S5 probe standup ids>")


def record_s5_windows(*, rs: Path, plan: dict, standup_ids: Sequence[str], accepted_get, read_text, replace: bool = False,
                      smoke_id: str = S5_SMOKE_ID) -> dict:
    """DERIVE the window measurements from the S5 probe rows' accepted attempts (`lp.derive_s5_window_measurements`: the LIVE
    accepted Dict, the producer's bound `audits/s5_window/...` record per probe child, both probe cells + a parent per leg, zero clamped calls on
    complete stores), validate each exactly as the FINAL plan's gate will (`lp.s5_smokes_add_window` -> `window_measurement_problems` + the
    execution wave) and merge them into `S5.multiturn[]`; the derivation record (rows, accepted attempts, root facts, every source's sha) is kept
    under `S5.window_derivation`. Nothing the operator types becomes a measurement."""
    doc = load_smokes(rs)
    execution_wave = execution_wave_for(rs, smoke_id)
    lp.s5_block_bind(doc, W=plan["W"], execution_wave_uuid=execution_wave, production_wave_uuid=plan["wave_uuid"], launch_plan_sha=plan["launch_plan_sha"])
    # the recorder RE-DERIVES THE COMPLETE UNION of the rows already derived and the rows requested, sorted + deduplicated --
    # exactly the set and order certify's fresh derivation produces -- and REBUILDS `S5.multiturn[]` from it; it never concatenates
    # source lists (per-leg recording in either order and a partial --replace both certify). A requested row already derived refuses
    # without --replace.
    prior = doc["S5"].get("window_derivation")
    prior_ids = set(str(x) for x in prior.get("standup_ids") or []) if isinstance(prior, Mapping) and isinstance(prior.get("standup_ids"), list) else set()
    requested = sorted({str(x) for x in standup_ids})
    if not requested:
        raise ValueError("window: name at least one S5 probe standup id")
    overlap = sorted(set(requested) & prior_ids)
    if overlap and not replace:
        raise ValueError(f"S5 window measurements are already derived for rows {overlap}; pass --replace to re-derive them (the whole set is re-derived either way)")
    union = sorted(prior_ids | set(requested))
    derivation, measurements = lp.derive_s5_window_measurements(rs=rs, plan=plan, standup_ids=union, accepted_get=accepted_get, read_text=read_text)
    if derivation["execution_wave_uuid"] != execution_wave:
        raise ValueError(f"the derivation's execution wave {derivation['execution_wave_uuid']} is not the S5 smoke's {execution_wave}")
    doc["S5"]["multiturn"] = []
    for m in measurements:
        lp.s5_smokes_add_window(doc, m, wave_uuid=execution_wave, replace=False)      # fresh records, one per (leg, arm) by construction
    doc["S5"]["window_derivation"] = derivation
    write_smokes(rs, doc)
    return doc


def _accepted_getter(plan: dict, smoke_id: str, rs: Path) -> lp.RecordingGetter:
    """The smoke wave's LIVE accepted Dict as a RECORDING getter: named by the smoke identity, read through the Dict API,
    every read recorded for the certificate. No snapshot exists -- an operator cannot select a superseded attempt into the record."""
    import seedrep_carry_over as co  # noqa: PLC0415
    if smoke_id == co.PRODUCTION_ACCEPTED_ID:
        # fj2: the certificate's carry-over block is held to THIS wave's (production) accepted Dict -- the carried keys must name the attempts
        import modal  # noqa: PLC0415
        d = modal.Dict.from_name(plan["dicts"]["accepted"], create_if_missing=False)
        return lp.RecordingGetter(plan["dicts"]["accepted"], d.get)
    wave = lp.smoke_execution_wave(rs, smoke_id)
    if wave is None:
        raise ValueError(f"no {lp.smoke_wave_path(rs, smoke_id)}: smoke {smoke_id} never ran")
    identity = lp.smoke_identity(plan, smoke_id, wave)
    import modal  # noqa: PLC0415
    d = modal.Dict.from_name(identity["dicts"]["accepted"], create_if_missing=False)
    return lp.RecordingGetter(identity["dicts"]["accepted"], d.get)


def _volume_reader(rs: Path):
    """THE evidence reader: `/vol/<rel>` through the Modal Volume API and NOTHING else -- no local pull, no
    operator directory (the former pull flag is gone: a local tree under the expected names is never consulted). Tests inject a reader through the
    Python seam (`main(argv, read_text=)` / the record functions' ``read_text``), which has no CLI surface."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import seedrep_watch as sw  # noqa: E402
    import modal  # noqa: PLC0415
    vol = modal.Volume.from_name(lp.MODAL_VOLUME_NAME, create_if_missing=False)

    def read(path: str):
        if str(path).startswith(lp.VOL_PATH + "/"):
            return sw.volume_read_text(vol, str(path))
        return None
    return read


def _lazy_volume_reader(rs: Path):
    """`_volume_reader(rs)` built on the first read (fc9): a record that needs no volume read never opens the Volume handle."""
    box: Dict[str, object] = {}

    def read(path: str):
        if "r" not in box:
            box["r"] = _volume_reader(rs)
        return box["r"](path)  # type: ignore[operator]
    return read


def record_smoke_pass(*, rs: Path, plan: dict, smoke_id: str, standup_ids: Sequence[str], accepted_get, read_text, draft: Optional[dict] = None) -> dict:
    """S1-S4: DERIVE the smoke's PASS record (`lp.derive_smoke_pass`: the accepted attempt per named row off the smoke
    wave's accepted Dict, its root facts off the volume, every check off the pointed output's content), validate it exactly as the certificate
    will (re-hashing every source), then write it under `smokes.json[<smoke_id>]` (write-once). Nothing the operator types becomes a verdict."""
    rec = lp.derive_smoke_pass(rs=rs, plan=plan, smoke_id=smoke_id, standup_ids=standup_ids, accepted_get=accepted_get, read_text=read_text)
    problems = lp.smoke_pass_problems(rec, smoke_id=smoke_id, plan=plan, execution_wave_uuid=lp.smoke_execution_wave(rs, smoke_id),
                                      draft_launch_plan_sha=(draft or {}).get("launch_plan_sha"), read_text=read_text, accepted_get=accepted_get)
    if problems:
        raise ValueError(f"the derived {smoke_id} record does not validate: {problems}")
    doc = load_smokes(rs)
    if smoke_id in doc:
        raise ValueError(f"smoke {smoke_id} already has a record in {Path(rs) / SMOKES_JSON}; a PASS is written once")
    doc[smoke_id] = rec
    write_smokes(rs, doc)
    return doc


def load_priors(specs: Sequence[str]) -> list:
    """The prior archives named on the command line (`RS/prior/<W-sha7>`, the carry-over's `load_prior_source` layout), in order, distinct."""
    import seedrep_carry_over as co  # noqa: PLC0415
    paths = [Path(x.strip()).expanduser() for x in specs if x.strip()]
    if not paths:
        raise ValueError("--prior names no archive")
    if len({str(x) for x in paths}) != len(paths):
        raise ValueError(f"--prior names an archive twice: {[str(x) for x in paths]}")
    return [co.load_prior_source(x) for x in paths]


def _dict_getters_factory(getters: Dict[str, lp.RecordingGetter], accepted_get_for_dict=None):
    """`accepted_get_for_dict(name) -> get` over the PRIOR smoke Dicts, one RECORDING getter per Dict name (the certificate carries the reads)."""
    def for_dict(name: str) -> lp.RecordingGetter:
        if name not in getters:
            if accepted_get_for_dict is not None:
                getters[name] = lp.RecordingGetter(name, accepted_get_for_dict(name))
            else:
                import modal  # noqa: PLC0415
                d = modal.Dict.from_name(name, create_if_missing=False)
                getters[name] = lp.RecordingGetter(name, d.get)
        return getters[name]
    return for_dict


def _default_reaudit(*, rs: Path, plan: dict, tree: Optional[Path]):
    """fc11: THIS tree's request-evidence re-audit over a prior root -- the carry-over's fj7 `seedrep_served.reaudit_carried_root` (pull the historical
    root, run this tree's `request_evidence_audit.py` with `--audited-at <this W>`, bind, upload under this wave's carried evidence root)."""
    import seedrep_served as srv  # noqa: PLC0415

    def reaudit(**kw):
        return srv.reaudit_carried_root(rs=Path(rs), this_sha=plan["W"], **({"tree": Path(tree)} if tree else {}), **kw)
    return reaudit


def record_smoke_pass_from_priors(*, rs: Path, plan: dict, smoke_id: str, standup_ids: Sequence[str], priors: Sequence[Mapping[str, Any]], accepted_get_for_dict,
                                  read_text, reaudit, draft: Optional[dict] = None) -> dict:
    """fc11: DERIVE the S1-S3 PASS record from PRIOR trees' accepted smoke attempts (`lp.derive_smoke_pass_from_priors`: each row's attempt off the
    PRIOR smoke Dict, its identity bound to this plan's row, this tree's re-audit over the historical stores), validate it exactly as the certificate
    will, write it once under `smokes.json[<smoke_id>]`."""
    rec = lp.derive_smoke_pass_from_priors(rs=rs, plan=plan, smoke_id=smoke_id, standup_ids=standup_ids, priors=priors, accepted_get_for_dict=accepted_get_for_dict,
                                           read_text=read_text, reaudit=reaudit)
    problems = lp.smoke_pass_problems(rec, smoke_id=smoke_id, plan=plan, execution_wave_uuid=lp.smoke_execution_wave(rs, smoke_id),
                                      draft_launch_plan_sha=(draft or {}).get("launch_plan_sha"), read_text=read_text, accepted_get_for_dict=accepted_get_for_dict)
    if problems:
        raise ValueError(f"the derived {smoke_id} record (from prior trees) does not validate: {problems}")
    doc = load_smokes(rs)
    if smoke_id in doc:
        raise ValueError(f"smoke {smoke_id} already has a record in {Path(rs) / SMOKES_JSON}; a PASS is written once")
    doc[smoke_id] = rec
    write_smokes(rs, doc)
    return doc


def _recorded_reaudit_for(rec: Mapping[str, Any], smoke_id: str):
    """At certify a prior-tree record's re-audit is NOT re-run (it would re-upload evidence); the RECORDED per-row re-audit record is handed back to the
    re-derivation, which re-reads its evidence root and recomputes every verdict; a recorded record for another root / attempt refuses by name."""
    rows = rec.get("rows") if isinstance(rec.get("rows"), Mapping) else {}

    def reaudit(*, prev_row, this_row, prior, aid, root, this_plan):
        r = rows.get(this_row["standup_id"]) if isinstance(rows, Mapping) else None
        ra = r.get("reaudit") if isinstance(r, Mapping) else None
        if not isinstance(ra, Mapping):
            raise ValueError(f"smokes.json[{smoke_id}] row {this_row['standup_id']} records no re-audit")
        if r.get("accepted_attempt") != aid or ra.get("store_root") != root:
            raise ValueError(f"smokes.json[{smoke_id}] row {this_row['standup_id']}: the recorded re-audit is of attempt {r.get('accepted_attempt')!r} at "
                             f"{ra.get('store_root')!r}, the prior Dict names {aid!r} at {root}")
        return dict(ra)
    return reaudit


def record_w_gate(*, rs: Path, plan: dict, baseline_sha: str, read_text) -> Path:
    """The W gate: DERIVED from the two merged CPU-suite summaries (`lp.derive_w_gate` hashes them and computes
    `introduced` itself); refused when W introduces a node; written once at RS/w_gate.json."""
    rec = lp.derive_w_gate(W=plan["W"], baseline_sha=baseline_sha, read_text=read_text)
    problems = lp.w_gate_problems(rec, W=plan["W"], read_text=read_text)
    if problems:
        raise ValueError(f"refusing to record the W gate: {problems}")
    out = Path(rs) / W_GATE_JSON
    if out.exists():
        raise ValueError(f"{out} exists; the W gate verdict is written once per W")
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return out


def write_decode_proof(*, rs: Path, tree: Path) -> Path:
    proof = lp.decode_ref_proof(Path(tree))
    problems = lp.decode_proof_problems(proof, registry_sha256=proof["registry_sha256"])
    out = Path(rs) / DECODE_PROOF_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proof, indent=1) + "\n")
    if problems:
        raise ValueError(f"the tree's registry does not resolve every battery row by exact ref (proof written to {out} as evidence): {problems}")
    return out


def _load_json(p: Optional[Path]):
    return json.loads(Path(p).read_text()) if p and Path(p).is_file() else None


def _rederived(rec: dict, fresh: dict, *, who: str, ignore: Sequence[str] = ("derived_utc",)) -> dict:
    a = {k: v for k, v in fresh.items() if k not in ignore}
    b = {k: v for k, v in rec.items() if k not in ignore}
    if a != b:
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise ValueError(f"{who} is not what the wave derives today (differing {diff}) -- a recorded document is never certified as such")
    return fresh


def rederive_smokes(*, rs: Path, plan: dict, draft: Optional[dict], smokes_doc: Any, accepted_get_for, read_text, accepted_get_for_dict=None) -> dict:
    """`certify` accepts NO smoke documents -- every S1-S4 record of `RS/smokes.json` is RE-DERIVED from its own rows and
    pointers (the smoke wave's accepted Dict as it stands, the attempt roots off the volume, the check outputs' content) and must equal the record
    it replaces; the certificate carries the re-derivation. The S5 block's WINDOWS are re-derived too (`rederive_s5_windows`); its
    references / bars are carried as recorded (bound by their wrapper and re-validated on content by `certificate_problems`)."""
    if not isinstance(smokes_doc, Mapping):
        raise ValueError(f"{Path(rs) / SMOKES_JSON} is missing or not an object -- no smoke has been recorded")
    out = dict(smokes_doc)
    admissible = {plan.get("launch_plan_sha")} | ({draft.get("launch_plan_sha")} if draft else set())
    for sid in lp.SMOKE_PASS_IDS:
        rec = smokes_doc.get(sid)
        if rec is None:
            continue
        if not isinstance(rec, Mapping) or not isinstance(rec.get("checks"), Mapping) or not isinstance(rec.get("standup_ids"), list):
            raise ValueError(f"smokes.json[{sid}] is not a derived PASS record (no rows / checks to re-derive from)")
        target = plan if rec.get("launch_plan_sha") == plan.get("launch_plan_sha") or not draft else draft
        if rec.get("launch_plan_sha") not in admissible:
            raise ValueError(f"smokes.json[{sid}] targets launch_plan_sha {rec.get('launch_plan_sha')!r}, neither this plan's nor its draft's")
        if lp.is_prior_smoke_pass(rec):
            # fc11: a record drawing on PRIOR trees is re-derived through the archives it names (must still be at hand), the PRIOR smoke Dicts as they
            # stand, and its RECORDED re-audit's evidence root re-read by this tree's readers; it must equal the record
            if accepted_get_for_dict is None:
                raise ValueError(f"smokes.json[{sid}] draws on prior trees {sorted(set((rec.get('measured_under_W') or {}).values()))} -- certify needs the prior Dict reader")
            specs = [str(p.get("path")) for p in (rec.get("priors") or []) if isinstance(p, Mapping)]
            gone = [x for x in specs if not Path(x).exists()]
            if gone:
                raise ValueError(f"smokes.json[{sid}]: the prior archive(s) {gone} it derives from are not at hand -- a prior-tree record cannot be re-derived without them")
            fresh = lp.derive_smoke_pass_from_priors(rs=rs, plan=target, smoke_id=sid, standup_ids=rec["standup_ids"], priors=load_priors(specs),
                                                     accepted_get_for_dict=accepted_get_for_dict, read_text=read_text, reaudit=_recorded_reaudit_for(rec, sid))
            out[sid] = _rederived(dict(rec), fresh, who=f"smokes.json[{sid}]")
            continue
        fresh = lp.derive_smoke_pass(rs=rs, plan=target, smoke_id=sid, standup_ids=rec["standup_ids"], accepted_get=accepted_get_for(sid), read_text=read_text)
        out[sid] = _rederived(dict(rec), fresh, who=f"smokes.json[{sid}]")
    s5 = smokes_doc.get("S5")
    if isinstance(s5, Mapping) and (s5.get("multiturn") or s5.get("window_derivation") is not None):
        # the windows are re-derived from the probe attempts too (the references / bars stay as recorded -- see the report's gaps)
        out["S5"] = rederive_s5_windows(rs=rs, plan=plan, draft=draft, s5=s5, accepted_get=accepted_get_for(S5_SMOKE_ID), read_text=read_text)
    return out


def admit_s4_waiver(*, rs: Path, plan: dict, smokes_doc: dict, read_text) -> dict:
    """`RS/s4_waiver.json` -- the documented S4 waiver -- stands in for the S4 PASS record when the multi-turn smokes never ran to
    completion. Validated by `lp.s4_waiver_problems` (the closed schema; W = this plan's; every named attempt root's RESULT re-read off the
    VOLUME and held to the recorded line) and recorded as `smokes.S4 = {status: WAIVED, ...}` (`lp.smoke_waiver_entry`). Refuses when both a
    derived S4 PASS record and the waiver file exist (one of them does not apply; nothing is picked silently); no file = nothing admitted,
    and `certificate_problems` then refuses the missing S4 record exactly as before."""
    rs = Path(rs)
    waiver_p = rs / lp.S4_WAIVER_FILE
    if not waiver_p.is_file():
        return smokes_doc
    rec = smokes_doc.get(lp.S4_WAIVER_SMOKE_ID)
    if isinstance(rec, Mapping) and rec.get("schema") == lp.SMOKE_PASS_SCHEMA:
        raise ValueError(f"both a derived {lp.S4_WAIVER_SMOKE_ID} PASS record ({SMOKES_JSON}) and {waiver_p} exist -- an S4 that PASSED needs no waiver; "
                         "remove the one that does not apply (nothing is picked silently)")
    try:
        doc = json.loads(waiver_p.read_text())
    except ValueError as e:
        raise ValueError(f"{waiver_p}: not JSON ({e})")
    probs = lp.s4_waiver_problems(doc, plan=plan, read_text=read_text)
    if probs:
        raise ValueError(f"{waiver_p}: not an admissible S4 waiver (D133): " + "; ".join(probs))
    out = dict(smokes_doc)
    out[lp.S4_WAIVER_SMOKE_ID] = lp.smoke_waiver_entry(doc, path=f"RS/{lp.S4_WAIVER_FILE}", file_sha256=lp.sha256_file(waiver_p))
    return out


def rederive_s5_windows(*, rs: Path, plan: dict, draft: Optional[dict], s5: Any, accepted_get, read_text) -> dict:
    """The S5 block's window measurements RE-DERIVED at certify the way S1-S4 are -- `lp.derive_s5_window_measurements` over the rows
    `S5.window_derivation` names (the LIVE accepted Dict, the producer's bound `audits/s5_window/...` files off the volume) must reproduce BOTH the
    derivation record and `S5.multiturn[]` exactly (per (leg, arm)); a block whose `multiturn[]` was hand-edited after `window --rows`, or that
    carries windows with no derivation at all, is refused by name. Returns the block with the fresh records in place of the recorded ones."""
    if not isinstance(s5, Mapping):
        raise ValueError("smokes.json[S5] is not an object")
    recorded = list(s5.get("multiturn") or [])
    der = s5.get("window_derivation")
    if not isinstance(der, Mapping) or not isinstance(der.get("standup_ids"), list) or not der.get("standup_ids"):
        raise ValueError("smokes.json[S5].multiturn[] carries no S5.window_derivation (the probe rows it was derived from) -- the windows were not written by "
                         "`seedrep_s5_record.py window --rows`; a hand-authored window record is never certified")
    admissible = {plan.get("launch_plan_sha")} | ({draft.get("launch_plan_sha")} if draft else set())
    if der.get("launch_plan_sha") not in admissible:
        raise ValueError(f"smokes.json[S5].window_derivation targets launch_plan_sha {der.get('launch_plan_sha')!r}, neither this plan's nor its draft's")
    target = plan if der.get("launch_plan_sha") == plan.get("launch_plan_sha") or not draft else draft
    fresh, measurements = lp.derive_s5_window_measurements(rs=rs, plan=target, standup_ids=[str(x) for x in der["standup_ids"]], accepted_get=accepted_get, read_text=read_text)
    _rederived(dict(der), fresh, who="smokes.json[S5].window_derivation")
    by_key = lambda ms: {(str(m.get("leg")), str(m.get("arm"))): m for m in ms if isinstance(m, Mapping)}  # noqa: E731
    got, want = by_key(recorded), by_key(measurements)
    if len(got) != len(recorded) or set(got) != set(want) or any(got[k] != want[k] for k in want):
        diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        raise ValueError(f"smokes.json[S5].multiturn[] is not what the probe attempts derive today (differing (leg, arm) {diff}) -- a window record edited "
                         "after `window --rows` is never certified")
    out = dict(s5)
    out["multiturn"] = measurements
    out["window_derivation"] = fresh
    return out


def rederive_w_gate(*, plan: dict, rec: Any, read_text) -> dict:
    """The W gate re-derived from the two summaries the record names: the summaries' own `sha` / `app` must be W's
    and the baseline's; the re-derivation must equal the record."""
    if not isinstance(rec, Mapping) or not isinstance(rec.get("baseline_sha"), str):
        raise ValueError(f"{W_GATE_JSON} is missing or not a derived W gate record (no baseline sha to re-derive from)")
    fresh = lp.derive_w_gate(W=plan["W"], baseline_sha=str(rec.get("baseline_sha")), read_text=read_text)      # the summaries' paths are DERIVED from the shas
    return _rederived(dict(rec), fresh, who=W_GATE_JSON)


parse_launch_set = lp.parse_launch_set     # fc10b: ONE parser for `--launch-set` (certify, the S0 gates, the FINAL generator)


def certify(*, rs: Path, plan_path: Path, draft_path: Path, tree: Path, s0: Optional[Path] = None, decode_proof: Optional[Path] = None,
            out: Optional[Path] = None, read_text=None, accepted_get_for=None, launch_set: Optional[Sequence[str]] = None, accepted_get_for_dict=None) -> Path:
    """Issue `RS/launch_certificate.json` (write-once, sealed) from the validated components; refuses naming every missing / failing one. The S1-S4
    records and the W gate are RE-DERIVED from `RS/smokes.json` / `RS/w_gate.json`'s own pointers (never ingested from a path)."""
    rs = Path(rs)
    plan = json.loads(Path(plan_path).read_text())
    draft = json.loads(Path(draft_path).read_text())
    reg = Path(tree) / lp.DECODE_REGISTRY_REL
    read_text = read_text or _volume_reader(rs)                          # the Volume API; never a local path
    getters: Dict[str, lp.RecordingGetter] = {}

    def live_for(sid: str):
        if sid not in getters:
            getters[sid] = accepted_get_for(sid) if accepted_get_for is not None else _accepted_getter(plan, sid, rs)
        return getters[sid]
    dict_getters: Dict[str, lp.RecordingGetter] = {}
    for_dict = _dict_getters_factory(dict_getters, accepted_get_for_dict)          # fc11: the PRIOR trees' smoke Dicts, read on demand, every read recorded
    smokes_doc = rederive_smokes(rs=rs, plan=plan, draft=draft, smokes_doc=load_smokes(rs), accepted_get_for=live_for, read_text=read_text, accepted_get_for_dict=for_dict)
    smokes_doc = admit_s4_waiver(rs=rs, plan=plan, smokes_doc=smokes_doc, read_text=read_text)
    w_gate = rederive_w_gate(plan=plan, rec=_load_json(rs / W_GATE_JSON), read_text=read_text)
    reads = {sid: (g.record() if isinstance(g, lp.RecordingGetter) else {"dict": None, "read_utc": None, "entries": {}, "digest": lp.accepted_reads_digest({})})
             for sid, g in getters.items()}
    for sid in lp.SMOKE_PASS_IDS:
        rec = smokes_doc.get(sid)
        if lp.is_prior_smoke_pass(rec):
            # fc11: the reads of the prior Dicts this record's rows resolved through, merged under their joined names (`prior_smoke_pass_dict_name`)
            names = sorted({str(r.get("accepted_dict")) for r in (rec.get("rows") or {}).values() if isinstance(r, Mapping)})
            entries: Dict[str, Any] = {}
            utcs = []
            for n in names:
                g = dict_getters.get(n)
                if g is not None:
                    entries.update(g.entries)
                    utcs.append(g.read_utc or "")
            reads[sid] = {"dict": lp.prior_smoke_pass_dict_name(rec), "read_utc": (max(utcs) if utcs else None), "entries": dict(sorted(entries.items())),
                          "digest": lp.accepted_reads_digest(entries)}
    cert = lp.build_certificate(plan=plan, plan_path=Path(plan_path), draft=draft, s0_record=_load_json(s0 or rs / "s0_gates.json"),
                                smokes_doc=smokes_doc, w_gate=w_gate, decode_proof=_load_json(decode_proof or rs / DECODE_PROOF_JSON), rs=rs,
                                registry_sha256=(lp.sha256_file(reg) if reg.is_file() else None), read_text=read_text, accepted_get_for=live_for, accepted_reads=reads,
                                launch_set=launch_set, accepted_get_for_dict=for_dict)
    dest = Path(out) if out else rs / lp.CERTIFICATE_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)          # IMMUTABLE: never overwritten; a new plan gets a new certificate elsewhere
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(cert, indent=1) + "\n")
    return dest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for kind in ("reference", "bars"):
        c = sub.add_parser(kind)
        c.add_argument("--rs", type=Path, required=True, help="the results root (RS/smokes.json lives here)")
        c.add_argument("--plan", type=Path, required=True, help="the wave's launch plan (its W and wave_uuid bind the record)")
        c.add_argument("--record", type=Path, required=True, help="the measured record, JSON")
        c.add_argument("--replace", action="store_true", help="supersede a record already present for the same window class / (leg, arm)")
        c.add_argument("--smoke-id", default=S5_SMOKE_ID, help="the smoke whose execution wave the record binds to (RS/smokes/<id>/wave.json; default S5)")
    c = sub.add_parser("window", help="DERIVE the D50 multi-turn window measurements from the S5 probe rows' accepted attempts (the producer's bound "
                                      "audits/s5_window/ records) and merge them into smokes.json S5.multiturn[]; --record is REFUSED")
    c.add_argument("--rs", type=Path, required=True, help="the results root (RS/smokes.json lives here)")
    c.add_argument("--plan", type=Path, required=True, help="the wave's launch plan (its W and wave_uuid bind the record)")
    c.add_argument("--rows", default=None, help="comma-separated standup ids of the S5 probe rows (their accepted attempts are resolved LIVE, never typed)")
    c.add_argument("--record", type=Path, default=None, help=argparse.SUPPRESS)          # REFUSED by name in main (W16s5): kept so the refusal is explicit, never argparse's
    c.add_argument("--replace", action="store_true", help="supersede a measurement already present for the same (leg, arm)")
    c.add_argument("--smoke-id", default=S5_SMOKE_ID, help="the smoke whose execution wave the records bind to (RS/smokes/<id>/wave.json; default S5)")
    c = sub.add_parser("pass", help="DERIVE an S1-S4 smoke PASS record from pointers (the smoke rows + the check outputs' paths) and write it once under smokes.json[<id>]")
    c.add_argument("--rs", type=Path, required=True)
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--draft", type=Path, default=None, help="the draft the smoke ran against (its launch_plan_sha is admissible on the record)")
    c.add_argument("--smoke-id", required=True, choices=list(lp.SMOKE_PASS_IDS))
    c.add_argument("--rows", required=True, help="comma-separated standup ids of the plan rows the smoke ran (their accepted attempts are resolved LIVE, never typed)")
    c.add_argument("--prior", default=None, help="fc11: comma-separated PRIOR archives (RS/prior/<W7>, the carry-over's layout) whose smoke Dicts hold the rows' accepted "
                                                 "attempts; each row is bound to this plan's by identity and RE-AUDITED by this tree (S1-S3 only; S4 is the D133 waiver)")
    c.add_argument("--tree", type=Path, default=None, help="with --prior: the W checkout whose request_evidence_audit.py re-audits (default: this script's repo)")
    c = sub.add_parser("w-gate", help="DERIVE the W gate verdict from the two merged CPU-suite summaries (introduced == [] by node id) -> RS/w_gate.json, write-once")
    c.add_argument("--rs", type=Path, required=True)
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--baseline-sha", required=True, help="the baseline gate's full sha; both summaries are read at /vol/cpu-suite/<sha>/summary.json (derived, never supplied)")
    c = sub.add_parser("decode-proof", help="resolve the battery's eight decode rows by exact ref through the tree's registry -> RS/decode_ref_proof.json")
    c.add_argument("--rs", type=Path, required=True)
    c.add_argument("--tree", type=Path, required=True)
    c = sub.add_parser("certify", help="issue RS/launch_certificate.json (sealed, write-once) from the validated components; an RS/s4_waiver.json "
                                        "(D133: the documented S4 waiver, closed schema, roots re-read off the volume) stands in for the S4 PASS record")
    c.add_argument("--rs", type=Path, required=True)
    c.add_argument("--plan", type=Path, required=True, help="RS/launch_plan.json (FINAL)")
    c.add_argument("--draft", type=Path, required=True, help="RS/launch_plan_draft.json (the smokes ran against it)")
    c.add_argument("--tree", type=Path, required=True, help="the W checkout (its decode registry bytes bind the proof)")
    c.add_argument("--s0", type=Path, default=None)
    c.add_argument("--decode-proof", type=Path, default=None)
    c.add_argument("--out", type=Path, default=None)
    c.add_argument("--launch-set", default=None,
                   help="fc9 / D146: the standup ids this certificate certifies for launch (comma-separated, or @<file>: one per line / a JSON list); the S5 "
                        "coverage is derived for THEIR window classes alone and --go refuses any row outside the set; omitted = every planned row")

    return ap


def main(argv=None, *, read_text=None, accepted_get_for=None, accepted_get_for_dict=None, reaudit=None) -> int:
    """``read_text`` / ``accepted_get_for`` / ``accepted_get_for_dict`` / ``reaudit`` are PYTHON-ONLY seams (tests): the CLI reads the Volume API and the LIVE
    Dicts, re-audits through `seedrep_served.reaudit_carried_root`, and offers no flag to do otherwise."""
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "pass" and a.prior:
            plan = json.loads(Path(a.plan).read_text())
            rows = [s_.strip() for s_ in a.rows.split(",") if s_.strip()]
            getters: Dict[str, lp.RecordingGetter] = {}
            doc = record_smoke_pass_from_priors(rs=a.rs, plan=plan, smoke_id=a.smoke_id, standup_ids=rows, priors=load_priors(a.prior.split(",")),
                                                accepted_get_for_dict=_dict_getters_factory(getters, accepted_get_for_dict), read_text=read_text or _volume_reader(a.rs),
                                                reaudit=reaudit or _default_reaudit(rs=a.rs, plan=plan, tree=a.tree), draft=_load_json(a.draft))
            rec = doc[a.smoke_id]
            print(f"[s5-record] smoke {a.smoke_id} PASS DERIVED FROM PRIOR TREES {sorted({w[:7] for w in rec['measured_under_W'].values()})} -> {Path(a.rs) / SMOKES_JSON}: "
                  f"rows {rec['accepted']}, checks {sorted(rec['checks'])}, re-audited by {rec['reaudited_by'][:12]}")
            return 0
        if a.cmd == "pass":
            plan = json.loads(Path(a.plan).read_text())
            doc = record_smoke_pass(rs=a.rs, plan=plan, smoke_id=a.smoke_id, standup_ids=[s.strip() for s in a.rows.split(",") if s.strip()],
                                    accepted_get=(accepted_get_for(a.smoke_id) if accepted_get_for is not None else _accepted_getter(plan, a.smoke_id, a.rs)),
                                    read_text=read_text or _volume_reader(a.rs), draft=_load_json(a.draft))
            rec = doc[a.smoke_id]
            print(f"[s5-record] smoke {a.smoke_id} PASS DERIVED -> {Path(a.rs) / SMOKES_JSON}: rows {rec['accepted']}, checks {sorted(rec['checks'])}")
            return 0
        if a.cmd == "w-gate":
            plan = json.loads(Path(a.plan).read_text())
            out = record_w_gate(rs=a.rs, plan=plan, baseline_sha=a.baseline_sha, read_text=read_text or _volume_reader(a.rs))
            print(f"[s5-record] W gate verdict DERIVED -> {out}")
            return 0
        if a.cmd == "decode-proof":
            out = write_decode_proof(rs=a.rs, tree=a.tree)
            print(f"[s5-record] decode-ref proof (all {len(lp.BATTERY_DECODE_REFS)} rows by exact ref) -> {out}")
            return 0
        if a.cmd == "certify":
            out = certify(rs=a.rs, plan_path=a.plan, draft_path=a.draft, tree=a.tree, s0=a.s0, decode_proof=a.decode_proof, out=a.out,
                          read_text=read_text, accepted_get_for=accepted_get_for, launch_set=parse_launch_set(a.launch_set), accepted_get_for_dict=accepted_get_for_dict)
            print(f"[s5-record] LAUNCH CERTIFICATE issued -> {out} (sealed; --go / --legs / --reattempt consume it and nothing else)")
            return 0
        if a.cmd == "window":
            if a.record is not None:
                raise ValueError(OPERATOR_RECORD_REFUSAL)
            if not a.rows:
                raise ValueError("window: --rows <the S5 probe standup ids> is required (the measurements are derived from their accepted attempts)")
            plan = json.loads(Path(a.plan).read_text())
            rows = [s.strip() for s in a.rows.split(",") if s.strip()]
            doc = record_s5_windows(rs=a.rs, plan=plan, standup_ids=rows, replace=a.replace, smoke_id=a.smoke_id,
                                    accepted_get=(accepted_get_for(a.smoke_id) if accepted_get_for is not None else _accepted_getter(plan, a.smoke_id, a.rs)),
                                    read_text=read_text or _volume_reader(a.rs))
            block = doc["S5"]
            print(f"[s5-record] window measurements DERIVED -> {Path(a.rs) / SMOKES_JSON}: {len(block.get('multiturn') or [])} record(s) "
                  f"{[(m['leg'], m['arm'], m['context_max'], m['clamped_calls']) for m in block.get('multiturn') or []]}; execution wave {block.get('execution_wave_uuid')}")
            return 0
        plan = json.loads(Path(a.plan).read_text())
        rec = json.loads(Path(a.record).read_text())
        doc = record_s5(a.cmd, rs=a.rs, plan=plan, record=rec, replace=a.replace, smoke_id=a.smoke_id, read_text=read_text)
    except (ValueError, FileExistsError) as e:
        print(f"[s5-record] REFUSED: {e}", file=sys.stderr)
        return 2
    block = doc["S5"]
    print(f"[s5-record] {a.cmd} recorded -> {Path(a.rs) / SMOKES_JSON}: {len(block.get('references') or [])} reference(s), "
          f"{len(block.get('multiturn') or [])} window measurement(s), bars_met={((block.get('bars') or {}).get('bars_met'))!r}; "
          f"execution wave {block.get('execution_wave_uuid')} -> production wave {block.get('production_wave_uuid')} plan {block.get('launch_plan_sha')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
