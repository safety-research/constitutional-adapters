"""registry_emission — ONE owner of the object writers' registry-emission EVIDENCE.

A published artefact (its ``COMPLETE`` marker, or a derived object's ``derived_meta.json``) is the publication event, NOT
the registry record. Every producer that emits an ``object`` row — ``scripts/train_two_si_objects.py`` (``publish_cell``),
``scripts/arditi/fit_refusal_direction.py``, ``scripts/sdfbase/lora_subtract.py``, ``scripts/sdfbase/psr_residual_distill.py``
— leaves, beside the artefact, after EVERY attempt:

* ``registry_emission.json`` = ``{object_id, status, utc, writer, …}`` with ``status`` ``emitted`` (the record paths the
  hook wrote, the campaign, the spool), ``not_emitted`` (emission off on this host / the registry package absent / a
  producer's stated reason — never silent) or ``failed`` (the error), and
* the bare ``REGISTRY_EMIT_FAILED`` marker on ``failed`` (a later success removes it) — so a shell
  (``reproduce/04_fit_objects.sh``) reads "emitted, waiting for the fold" / "never emitted" / "failed" over ssh with
  ``[ -s ]``, and a producer exits :data:`EMIT_FAILED_RC` on a failure with the artefact intact.

A resume over a KEPT artefact goes through :func:`ensure_emitted`: the evidence is verified (keyed by the artefact's
object id), else the producer spool is RECONCILED for BOTH records the hook writes — the ``object`` row and its
``artifact`` row — on every non-verifying status (absent, stale, ``failed``, ``not_emitted``): both present → evidence,
nothing emitted; one present → ONLY the missing record is replayed (``emit_component``); neither → the whole hook.
Never by rebuilding the artefact, never a duplicate native row (the hook emits ``object`` then ``artifact`` sequentially
and its guard returns None when the SECOND write fails, so ``failed`` evidence can sit beside a spool that already holds
the object row). Nothing here raises into the workload; nothing here is silent.

Loaded by file path by the scripts (``scripts/sdfbase`` is not on ``sys.path`` under ``python scripts/<x>.py``), the way
``tune_arm.py`` loads ``recon_clip``. ``train_two_si_objects._registry_publish`` keeps an INLINE copy of the evidence
writer because that function is pinned self-contained (tests exec its source alone); its literals are pinned equal to
:data:`EMISSION_NAME` / :data:`EMIT_FAILED_NAME` by ``tests/test_registry_emission_module.py``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import functools
from typing import Callable, Optional

EMISSION_NAME = "registry_emission.json"
EMIT_FAILED_NAME = "REGISTRY_EMIT_FAILED"
#: the producer's exit code when its emission FAILED and the artefact stands (0 = published + recorded / off by config)
EMIT_FAILED_RC = 3
STATUSES = ("emitted", "not_emitted", "failed")
#: the two records ``hooks.emit_object_from_sidecar`` writes for one object, in its write order
COMPONENTS = ("object", "artifact")
COMPLETE_NAME = "COMPLETE"
#: The emission CONTRACT the evidence attests to. Evidence written under an older contract is not terminal: the record it points
#: at may be one the registry refuses (f680858bc: the derived emitters' fit_objective `ce` died at the pairing CHECK with the
#: evidence reading "emitted"), or one filed under a campaign the job no longer states. Bump when the hook's record contract
#: changes in a way that makes earlier records inadmissible or wrongly framed; every kept-object replay then re-emits once.
#:   1 — (implicit; evidence without a `contract` field) records before 2026-10-02
#:   2 — stated campaign; derived records with fit_objective not_applicable + derivation_params + fit_writer (emitter-fix, f680858bc)
EVIDENCE_CONTRACT = "2"
#: the env var the front door's job env sets to the records' campaign (submit_twin.job_env / lib/slurm.sh slurm_job_env:
#: SISTEER_REGISTRY_CAMPAIGN, default sdfbase_full22 — the paper frame's); a manual re-emission sets it explicitly
CAMPAIGN_ENV = "REGISTRY_CAMPAIGN"


def utc_now() -> str:
    """THE registry spelling (records.utc_now, milliseconds); the plain second form only without the package."""
    try:
        from sisteer.registry.records import utc_now as _utc
        return _utc()
    except ImportError:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class CampaignUnset(RuntimeError):
    """No campaign for a record: ``REGISTRY_CAMPAIGN`` unset and no parent evidence names one, or the two disagree. A
    writer REFUSES here (the producer records ``failed`` evidence) instead of filing under a module default — until
    2026-10-02 the default ``two_si`` filed a front-door model's fit objects outside the paper frame's campaign
    (``sdfbase_full22``, which its 600 runs carry), and because the ``object`` table is keyed by ``object_id`` ALONE a
    default-campaign record for an adapter-filed object would UPSERT that row's campaign out of the frame
    (stage-E ruling 2026-10-02, the registry owner's hazard note)."""


def campaign_from_env(default: Optional[str] = None) -> str:
    """``REGISTRY_CAMPAIGN`` when set; else ``default`` when a caller passes one EXPLICITLY; else :class:`CampaignUnset`.
    There is no module default any more (the class docstring says why)."""
    v = os.environ.get(CAMPAIGN_ENV)
    if v:
        return v
    if default is not None:
        return default
    raise CampaignUnset(
        f"{CAMPAIGN_ENV} is unset — a record's campaign is the writer's statement, never a module default: the front "
        f"door's job env sets it (SISTEER_REGISTRY_CAMPAIGN, default sdfbase_full22); a manual re-emission runs with "
        f"{CAMPAIGN_ENV}=<campaign> set explicitly")


def emission_on() -> bool:
    """The hooks' one switch (``hooks.emission_enabled``); False when the registry package is absent."""
    try:
        from sisteer.registry import hooks as _rh
    except ImportError:
        return False
    return bool(_rh.emission_enabled(None))


def campaign_for_derived_or_off(meta) -> Optional[str]:
    """:func:`campaign_for_derived`, except that with registry emission OFF on this host an unresolvable campaign is None, not a
    refusal: off is a silent no-op (the hook writes nothing and the producer records ``not_emitted``), so it needs no campaign.
    With emission ON the refusal stands (the producer records ``failed`` evidence)."""
    try:
        return campaign_for_derived(meta)
    except CampaignUnset:
        if emission_on():
            raise
        return None


def parents_recorded_campaigns(meta) -> dict:
    """``{parent_pos: campaign, parent_neg: campaign}`` from each parent's OWN emission evidence
    (``<parent cell>/registry_emission.json``), reached through the ``object_path`` the derived sidecar records for it;
    None for a parent with no path or no evidence."""
    out = {}
    for role in ("parent_pos", "parent_neg"):
        blk = (meta or {}).get(role) if isinstance(meta, dict) else None
        path = blk.get("object_path") if isinstance(blk, dict) else None
        ev = read_evidence(Path(str(path)).parent) if path else None
        c = ev.get("campaign") if isinstance(ev, dict) else None
        out[role] = str(c) if c else None
    return out


def campaign_for_derived(meta) -> str:
    """A DERIVED object's campaign: its parents' recorded one (both parents' evidence must agree), an explicitly set
    ``REGISTRY_CAMPAIGN`` only when it agrees with them (or when no parent carries evidence), else :class:`CampaignUnset`
    naming what disagreed. A derived object never carries a campaign its parents do not (the paper frame joins them)."""
    recorded = {r: c for r, c in parents_recorded_campaigns(meta).items() if c}
    env = os.environ.get(CAMPAIGN_ENV) or None
    distinct = sorted(set(recorded.values()))
    if len(distinct) > 1:
        raise CampaignUnset(f"the parents' recorded campaigns disagree ({recorded}) — re-emit the parents under ONE campaign "
                            "before deriving from them")
    if distinct:
        c = distinct[0]
        if env and env != c:
            raise CampaignUnset(f"{CAMPAIGN_ENV}={env!r} but the parents' recorded campaign is {c!r} ({recorded}) — a derived "
                                f"object carries its parents' campaign: re-emit the parents under {env!r} first, or unset "
                                f"{CAMPAIGN_ENV}")
        return c
    if env:
        return env
    raise CampaignUnset(f"no parent carries emission evidence naming a campaign (parents: {parents_recorded_campaigns(meta)}) and "
                        f"{CAMPAIGN_ENV} is unset — set it explicitly for this emission")


def json_field(path, key: str) -> Optional[str]:
    """``key`` of a JSON file (a COMPLETE marker's ``object_sha256``, a sidecar's), or None when absent / unparseable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    v = doc.get(key) if isinstance(doc, dict) else None
    return str(v) if v else None


def marker_object_id(out_dir, *, marker: str = COMPLETE_NAME, key: str = "object_sha256") -> Optional[str]:
    return json_field(Path(out_dir) / marker, key)


def write_evidence(out_dir, status: str, *, object_id: Optional[str], writer: str, **more) -> dict:
    """The evidence beside the artefact, atomically; ``failed`` also writes the marker, any other status removes it."""
    if status not in STATUSES:
        raise ValueError(f"registry_emission: status {status!r} is not one of {STATUSES}")
    out_dir = Path(out_dir)
    ev = {"object_id": object_id, "status": status, "utc": utc_now(), "writer": writer, "contract": EVIDENCE_CONTRACT, **more}
    text = json.dumps(ev, indent=1, sort_keys=True, default=str) + "\n"
    tmp = out_dir / (EMISSION_NAME + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_dir / EMISSION_NAME)
    failed = out_dir / EMIT_FAILED_NAME
    if status == "failed":
        failed.write_text(text, encoding="utf-8")
    elif failed.exists():
        failed.unlink()
    return ev


def read_evidence(out_dir) -> Optional[dict]:
    p = Path(out_dir) / EMISSION_NAME
    if not p.is_file():
        return None
    try:
        ev = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return ev if isinstance(ev, dict) else None


def evidence_verifies(ev: Optional[dict], object_id: Optional[str], campaign: Optional[str] = None) -> bool:
    """Terminal evidence: ``emitted`` for exactly this object id, under the CURRENT emission contract, and — when the caller
    states the campaign — for that campaign. ``not_emitted`` and ``failed`` are never terminal (emission may have been enabled /
    repaired since); nor is evidence for another object id (a superseded artefact); nor evidence from an older contract (its
    record may be one the registry refused — V's `ce` object row, 2026-10-02 — an object-id match proves neither that the record
    was admitted nor that it is the one wanted now); nor evidence under another campaign (the parents filed under the emitters'
    old `two_si` default while the job now states sdfbase_full22)."""
    if not (ev and object_id and ev.get("status") == "emitted" and ev.get("object_id") == object_id):
        return False
    if str(ev.get("contract") or "1") != EVIDENCE_CONTRACT:
        return False
    if campaign is not None and ev.get("campaign") != campaign:
        return False
    return True


def evidence_mismatch(ev: Optional[dict], object_id: Optional[str], campaign: Optional[str] = None) -> str:
    """Why :func:`evidence_verifies` said no, for the replay's log line."""
    if ev is None:
        return "no emission evidence"
    if ev.get("status") != "emitted":
        return f"evidence status {ev.get('status')!r}"
    if ev.get("object_id") != object_id:
        return "evidence for another object id"
    if str(ev.get("contract") or "1") != EVIDENCE_CONTRACT:
        return f"evidence from emission contract {ev.get('contract') or '1'} (current {EVIDENCE_CONTRACT}: its record may be inadmissible)"
    if campaign is not None and ev.get("campaign") != campaign:
        return f"evidence under campaign {ev.get('campaign')!r}, this run states {campaign!r}"
    return "evidence verifies"


def _record_is_component(rec: dict, component: str, object_id: str) -> bool:
    """Whether a spool record is THIS object's ``object`` row (payload.object_id) or its ``artifact`` row
    (payload.artifact_id — the hook's artifact_id IS the object sha — or payload.extra.object_id)."""
    if rec.get("record_type") != component:
        return False
    payload = rec.get("payload") or {}
    if component == "object":
        return payload.get("object_id") == object_id
    return payload.get("artifact_id") == object_id or (payload.get("extra") or {}).get("object_id") == object_id


def spool_records_for(object_id: str, campaign: str) -> dict:
    """Record files already in THIS host's producer spool carrying each of this object's records — the directory
    ``hooks.emit_object_from_sidecar`` writes, ``<spool>/<campaign>/object__<sha16>/`` — as
    ``{"object": [paths], "artifact": [paths]}`` (both [] when the spool / directory is absent or the registry
    package is not importable). The spool is never drained by ``push_records.sh`` (it rsyncs), so presence here is a
    durable fact about what was written."""
    found: dict = {c: [] for c in COMPONENTS}
    found["refused"] = []                      # (file, field, reason) for this object's records the registry would REFUSE at ingest
    if not object_id or not campaign:          # no campaign (emission off, nothing resolvable): nothing to scan for
        return found
    try:
        from sisteer.registry import hooks as _rh
        from sisteer.registry import records as _R
        from sisteer.registry import validate as _VAL
        from sisteer.registry.identity import run_dir_name
    except ImportError:
        return found
    spool = _rh.resolve_spool()
    if not spool:
        return found
    d = Path(spool) / campaign / run_dir_name(f"object__{object_id[:16]}")
    if not d.is_dir():
        return found
    for f in _R.iter_record_files(str(d)):
        try:
            for _ln, rec, _sha in _R.read_record_file(f, compute_sha=False):
                for c in COMPONENTS:
                    if _record_is_component(rec, c, object_id) and f not in found[c]:
                        # ADMISSION-AWARE presence: a spooled record the validator refuses — the pairing
                        # CHECK's mirror first of all — is not a record the registry holds; it counts as absent so the replay
                        # re-states it (the fold is by natural key: a re-statement, never a duplicate row)
                        v = _VAL.validate_record(rec, f)
                        if isinstance(v, _VAL.Validated):
                            found[c].append(f)
                        else:
                            found["refused"].append((f, getattr(v, "field", None), getattr(v, "reason", str(v))))
        except (ValueError, OSError):
            continue
    return found


def spool_object_records(object_id: str, campaign: str) -> list[str]:
    """The ``object`` row's record files (:func:`spool_records_for`'s ``object`` list) — the fallback evidence for an
    artefact published before the evidence file existed. Kept for callers that ask only about the native row."""
    return spool_records_for(object_id, campaign)["object"]


def reconcile_spool(object_id: Optional[str], campaign: str) -> dict:
    """The spool's account of this object: ``state`` ``complete`` (object AND artifact rows present), ``partial``
    (exactly one of them) or ``absent``; ``present`` = :func:`spool_records_for`; ``missing`` = the components without a
    record, in the hook's write order. This is read on EVERY non-verifying evidence status before anything is emitted,
    so a failure between the two writes never costs a duplicate object row."""
    present = spool_records_for(object_id, campaign) if object_id else {c: [] for c in COMPONENTS}
    missing = [c for c in COMPONENTS if not present.get(c)]
    state = "complete" if not missing else ("absent" if len(missing) == len(COMPONENTS) else "partial")
    return {"state": state, "present": {c: present.get(c, []) for c in COMPONENTS}, "missing": missing,
            "records": [f for c in COMPONENTS for f in present.get(c, [])], "refused": list(present.get("refused") or [])}


def emit_sidecar_component(component: str, obj_dir, *, campaign: str, script: str, session: Optional[str] = None,
                           registry_emit: Optional[bool] = None, spool: Optional[str] = None,
                           push_script: Optional[str] = None, **hook_kwargs) -> list:
    """ONE of the two records ``hooks.emit_object_from_sidecar`` writes for the object under ``obj_dir`` — the same
    ``hooks.object_from_sidecar`` read, the same ``RecordWriter`` directory, only ``component`` written — for the
    partial-replay branch of :func:`ensure_emitted` (the hook writes both rows unconditionally, so replaying "only
    the missing one" needs a per-component writer with the hook's exact inputs).
    Takes the hook's keyword arguments verbatim (:func:`component_emitter` gives it the hook's call signature); returns
    the paths written ([] when emission is off, the hook's own contract); raises on failure — every caller runs it
    under :func:`emit_with_evidence`, which records the error as ``failed``."""
    if component not in COMPONENTS:
        raise ValueError(f"registry_emission: component {component!r} is not one of {COMPONENTS}")
    from sisteer.registry import emit as _emit
    from sisteer.registry import hooks as _rh
    if not _rh.emission_enabled(registry_emit):
        return []
    spool_path = _rh.resolve_spool(spool)
    if not spool_path:
        _rh.notice(f"{script}: emission on but no spool resolves; {component} record not written")
        return []
    producer = _emit.producer_block(script, session=session)
    payload, artifact = _rh.object_from_sidecar(str(obj_dir), campaign=campaign, producer=producer, **hook_kwargs)
    w = _emit.RecordWriter(spool_path, campaign, f"object__{str(payload['object_id'])[:16]}", producer,
                           push_script=push_script)
    path = w.emit_object(payload) if component == "object" else w.emit_artifact(artifact)
    w.push(final=True)
    return [path] if path else []


def component_emitter(component: str) -> Callable:
    """:func:`emit_sidecar_component` bound to ``component``, with the hook's call signature
    ``(obj_dir, *, campaign, script, …) -> paths`` — what a producer passes as its hook call's ``emitter=`` when the
    ensure path asks it to replay one component."""
    if component not in COMPONENTS:
        raise ValueError(f"registry_emission: component {component!r} is not one of {COMPONENTS}")
    return functools.partial(emit_sidecar_component, component)


def emit_with_evidence(out_dir, *, object_id: Optional[str], writer: str, campaign: str,
                       emit: Callable[[], object], what: str = "object") -> dict:
    """Run ``emit()`` — the producer's call into ``hooks.emit_object_from_sidecar`` — and turn its outcome into
    evidence: a non-empty list of record paths → ``emitted``; ``[]`` → ``not_emitted`` (emission off / no spool: the
    hook's own contract); ``None`` → ``failed`` (the guarded hook swallowed a failure: no record was written); an
    ImportError → ``not_emitted`` (registry package absent, stated); any other exception → ``failed``. Never raises,
    never silent: a failure is one stderr line naming the marker and the replay."""
    out_dir = Path(out_dir)
    try:
        paths = emit()
    except ImportError as e:
        print(f"[registry] {writer}: registry package unavailable — {what} not recorded ({e})", file=sys.stderr)
        return write_evidence(out_dir, "not_emitted", object_id=object_id, writer=writer,
                              reason=f"registry package unavailable ({e})")
    except Exception as e:  # noqa: BLE001 — a lost record never costs the artefact; it is RECORDED, not swallowed
        return _failed(out_dir, object_id, writer, what, f"{type(e).__name__}: {e}")
    if paths is None:
        return _failed(out_dir, object_id, writer, what,
                       "RuntimeError: the registry hook failed inside its guard (see the [registry] notice above)")
    if paths:
        spool = None
        try:
            from sisteer.registry import hooks as _rh
            spool = _rh.resolve_spool()
        except ImportError:
            pass
        return write_evidence(out_dir, "emitted", object_id=object_id, writer=writer,
                              records=[str(x) for x in paths], campaign=campaign, spool=spool)
    return write_evidence(out_dir, "not_emitted", object_id=object_id, writer=writer,
                          reason="registry emission is off on this host (REGISTRY_EMIT_DISABLED / REGISTRY_EMIT=0, "
                                 "or no spool resolves)")


def _failed(out_dir: Path, object_id, writer: str, what: str, err: str) -> dict:
    try:
        ev = write_evidence(out_dir, "failed", object_id=object_id, writer=writer, error=err)
    except Exception:  # noqa: BLE001 — the evidence write failing must not mask the notice
        ev = {"object_id": object_id, "status": "failed", "error": err, "writer": writer}
    print(f"[registry] {writer}: {what} not recorded ({err}) — {EMIT_FAILED_NAME} written beside the artefact; "
          "the artefact is kept and a re-run replays the record without rebuilding it", file=sys.stderr, flush=True)
    return ev


#: the fit-time provenance a replay must RESTATE rather than re-derive from its own context (hooks.FIT_PROVENANCE_KEYS; mirrored here
#: because this module imports nothing beyond the stdlib at import time)
FIT_PROVENANCE_KEYS = ("fit_code_sha", "fit_job_id", "fit_node_hostname", "fit_substrate", "fit_producer_session",
                       "fit_started_utc", "fit_finished_utc", "fit_walltime_s", "fit_gpu_type", "fit_gpu_count",
                       "fit_wandb_run_id", "fit_wandb_url")


def fit_provenance_from_prior_record(object_id: Optional[str], campaign: Optional[str]) -> Optional[dict]:
    """The object's PRIOR statement of its fit-time provenance: the FIT_PROVENANCE_KEYS of the earliest ``object`` record for this
    object id in ``<spool>/<campaign>/object__<sha16>/`` (the record the builder itself wrote — admissible or not: the provenance
    stands even when the record was refused for its objective) plus the matching artifact's ``created_utc`` as
    ``artifact_created_utc``. None when the spool / dir / record is absent. A kept-object REPLAY states this instead of its own
    context (the replaying job's checkout, SLURM_JOB_ID and host are not the fit's)."""
    if not object_id or not campaign:
        return None
    try:
        from sisteer.registry import hooks as _rh
        from sisteer.registry import records as _R
        from sisteer.registry.identity import run_dir_name
    except ImportError:
        return None
    spool = _rh.resolve_spool()
    if not spool:
        return None
    d = Path(spool) / campaign / run_dir_name(f"object__{object_id[:16]}")
    if not d.is_dir():
        return None
    prov: Optional[dict] = None
    created: Optional[str] = None
    for f in _R.iter_record_files(str(d)):
        try:
            for _ln, rec, _sha in _R.read_record_file(f, compute_sha=False):
                p = rec.get("payload") or {}
                if prov is None and rec.get("record_type") == "object" and p.get("object_id") == object_id:
                    prov = {k: p.get(k) for k in FIT_PROVENANCE_KEYS}
                    prov["_source"] = f"{f}#object"
                if created is None and rec.get("record_type") == "artifact" and p.get("artifact_id") == object_id and p.get("created_utc"):
                    created = str(p["created_utc"])
        except (ValueError, OSError):
            continue
    if prov is None:
        return None
    if created:
        prov["artifact_created_utc"] = created
    return prov


def failed(out_dir, object_id, writer: str, what: str, err: str) -> dict:
    """The producers' way to record a refusal that happened BEFORE the hook ran (a campaign that could not be resolved):
    ``failed`` evidence + the marker + the stderr line, exactly as a hook failure is recorded."""
    return _failed(Path(out_dir), object_id, writer, what, err)


def ensure_emitted(out_dir, *, object_id: Optional[str], writer: str, campaign: str,
                   emit: Callable[[], object], what: str = "object",
                   emit_component: Optional[Callable[[str], object]] = None) -> dict:
    """Emission as a REPLAYABLE, idempotent step over a KEPT artefact: verify the evidence (keyed by the artefact's
    object id); else RECONCILE the producer spool for both records (:func:`reconcile_spool`) — on EVERY non-verifying
    status, ``failed`` and ``not_emitted`` included: ``complete`` → evidence
    written (``verified_from: spool``), nothing emitted; ``partial`` → ONLY the missing record is replayed through
    ``emit_component(<component>)`` (the producer's hook call bound to :func:`component_emitter`), and evidence lists the
    spool's record with the replayed one (``replayed``); a producer that offers no ``emit_component`` gets ``failed``
    evidence naming the partial state — never a second object row; ``absent`` → re-emit the whole hook through
    :func:`emit_with_evidence`. Never rebuilds anything."""
    out_dir = Path(out_dir)
    ev = read_evidence(out_dir)
    if evidence_verifies(ev, object_id, campaign):
        return ev
    why = evidence_mismatch(ev, object_id, campaign)
    rec = reconcile_spool(object_id, campaign)
    if rec.get("refused"):
        why += "; the spool's record(s) for it would be REFUSED at ingest (" + "; ".join(
            f"{Path(f).name}: {fld} — {str(r)[:90]}" for f, fld, r in rec["refused"]) + ")"
    if rec["state"] == "complete":
        print(f"[registry] {out_dir}: {why}, but the {what}'s object + artifact records for {object_id[:16]}… are already "
              f"in the spool ({len(rec['records'])} file(s)) — evidence written, nothing re-emitted", flush=True)
        return write_evidence(out_dir, "emitted", object_id=object_id, writer=writer, records=rec["records"],
                              campaign=campaign, verified_from="spool")
    if rec["state"] == "partial":
        missing, = rec["missing"]
        present = [c for c in COMPONENTS if c not in rec["missing"]]
        if emit_component is None:
            return _failed(out_dir, object_id, writer, what,
                           f"PartialEmission: the spool holds this object's {present[0]} record but no {missing} record, "
                           f"and this writer offers no per-component replay — refusing to re-emit the whole hook "
                           f"(it would duplicate the {present[0]} row); replay the {missing} record with a writer that does")
        print(f"[registry] {out_dir}: {why}; the spool holds the {what}'s {present[0]} record for {object_id[:16]}… but no "
              f"{missing} record — replaying ONLY the {missing} record (nothing is rebuilt, no row is duplicated)", flush=True)
        ev2 = emit_with_evidence(out_dir, object_id=object_id, writer=writer, campaign=campaign,
                                 emit=lambda: emit_component(missing), what=f"{what} {missing} record")
        if ev2.get("status") == "emitted":
            ev2 = write_evidence(out_dir, "emitted", object_id=object_id, writer=writer,
                                 records=rec["records"] + list(ev2.get("records") or []), campaign=campaign,
                                 spool=ev2.get("spool"), verified_from="spool+replay", replayed=[missing])
        return ev2
    print(f"[registry] {out_dir}: kept artefact but {why} and nothing for it in the spool — re-emitting the object record "
          "(nothing is rebuilt)", flush=True)
    return emit_with_evidence(out_dir, object_id=object_id, writer=writer, campaign=campaign, emit=emit, what=what)


def load(from_file) -> "module":  # type: ignore[valid-type]
    """The one module instance for a script that is not on ``sys.path`` (``scripts/<x>.py`` run by path): already
    imported → that; else loaded from this file's location relative to ``from_file``'s ``scripts/`` root."""
    mod = sys.modules.get("registry_emission")
    if mod is not None:
        return mod
    import importlib.util
    here = Path(from_file).resolve()
    root = here.parent
    while root.name != "scripts" and root.parent != root:
        root = root.parent
    path = root / "sdfbase" / "registry_emission.py"
    spec = importlib.util.spec_from_file_location("registry_emission", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["registry_emission"] = m
    spec.loader.exec_module(m)
    return m
