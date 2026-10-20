"""The standup's serving-evidence record has an IDENTITY, the store carries a POINTER to it, and the grader FOLLOWS the
pointer (arditi-qwen-all-positions Phase A review, round 3, item 2).

`ar_common.sh ar_assert_serving_evidence native` persists the ONE evidence body's record (`f22_assert_native_evidence`:
endpoint + facts + problems) as `serving_evidence.json` beside each standup's endpoint. Until this module the grader
(`scripts/arditi/grade_arditi_static.py --serving-evidence <path>`) accepted ANY well-shaped record with `problems: []`
at a path supplied by hand: a foreign record (kernel `relu_gated_vector`, kind `psr`, response-only positions, layer 7)
was accepted and emitted as the raw run's `native_evidence`; SERVE_DIR is job-suffixed, neighbouring rungs and retries
have healthy records too, and the store carried no pointer to the evidence that produced its rows.

Three things bind them now, all here so the writer, the folder and the reader share ONE hash and ONE key set:

  * IDENTITY (`identity_for` / `stamp`): the record gains an `identity` block -- `evidence_sha256` (sha256 over the
    canonical JSON of the whole record with that one field absent, so the identity's own facts are covered too),
    `slurm_job_id`, `serve_dir`, `cell`, the served adapters' `residual_fingerprint`s (the body's fact), the pins'
    `object_sha256` for this rung, the `served_object_sha256` the bank view OBSERVED on the bytes it bound to that pin
    before the standup (round 5: the identity names the bytes that were served, and refuses to be stamped unless they
    are the pinned bytes), `expected_served_layer`. Written by the shell (`ar_serving_evidence_identity`) through this
    module, never by hand.
  * THE POINTER (`fold_into`): the drivers that write `served_target.json` (`scripts/jailbreak_run.py`, the rung
    driver `sampled/escalate_static.py`) fold `serving_evidence_sha256` + `serving_evidence_path` into the target's
    `evidence` dict -- the `low_dose_liveness_policy.fold_into` precedent -- when, and ONLY when, the launcher named the
    record in `SISTEER_SERVING_EVIDENCE_RECORD` (`ar_cell.sbatch`, after the standup's evidence check passed:
    `ar_bind_serving_evidence_record`). With the env unset nothing is folded and every production shard is
    byte-identical. A named record that is absent, unshaped or does not re-hash REFUSES before generation (the store
    untouched): a launcher that named a record meant it, and a silently un-bound store is the failure this exists to
    remove.
  * THE READER (`load_record`): re-hash the record at the pointer's path and hold it to the recorded sha and to the
    store's pointer; the grader then asserts the raw contract from the ARMS TABLE + PINS -- never from the record's
    self-description (`grade_arditi_static._native_serving_metadata`).

Everything here is stdlib. scripts/, never src/ (the serving tree's `serving_impl_commit` claim stays true)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

#: the launcher names the standup's persisted record here, AFTER its evidence check passed; unset = nothing is folded
ENV = "SISTEER_SERVING_EVIDENCE_RECORD"
#: the record's identity block and the one field the hash excludes (itself)
IDENTITY_KEY = "identity"
SHA_FIELD = "evidence_sha256"
#: the identity's REQUIRED fields (besides the sha)
IDENTITY_FIELDS: Tuple[str, ...] = ("slurm_job_id", "serve_dir", "cell", "residual_fingerprint", "object_sha256",
                                    "served_object_sha256", "expected_served_layer")
#: what the drivers fold into served_target.json's `evidence` dict
EVIDENCE_SHA_KEY = "serving_evidence_sha256"
EVIDENCE_PATH_KEY = "serving_evidence_path"


class BindingRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"serving_evidence_binding: REFUSED — {msg}")


def _is_sha256_hex(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(ch in "0123456789abcdef" for ch in v)


def canonical_sha256(rec: Mapping[str, Any]) -> str:
    """sha256 over the canonical JSON (sorted keys, no whitespace) of `rec` with `identity.evidence_sha256` absent --
    so the hash covers the body (endpoint, facts, problems) AND every other identity field."""
    body = {k: v for k, v in rec.items() if k != IDENTITY_KEY}
    ident = rec.get(IDENTITY_KEY)
    if isinstance(ident, Mapping):
        body[IDENTITY_KEY] = {k: v for k, v in ident.items() if k != SHA_FIELD}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def identity_for(body: Mapping[str, Any], *, slurm_job_id: str, serve_dir: str, cell: str, object_sha256: str,
                 served_object_sha256: str) -> Dict[str, Any]:
    """The identity block for ONE evidence body (the JSON `f22_assert_native_evidence` emitted, problems empty). The
    served adapters' fingerprints and the expected served layer are read off the body's OWN facts
    (`residual_adapter_fingerprints`, `expected_served_layer`); the job id, serve dir, cell, the pins' object sha and
    the sha the bank view OBSERVED on the served bytes come from the launcher. Refuses a body with problems, without
    the facts, an object sha that is not 64-hex, or (round 5) a served sha that is not 64-hex or is not the pinned
    sha -- the identity names the bytes that were served, and bytes other than the pinned rung's get no identity."""
    facts, problems = body.get("facts"), body.get("problems")
    if not isinstance(facts, Mapping) or not isinstance(problems, list):
        raise BindingRefusal("the evidence body is not the ONE body's record (facts + problems)")
    if problems:
        raise BindingRefusal(f"a refused evidence body gets no identity ({len(problems)} problem(s)); the refusal is persisted as is")
    fps = facts.get("residual_adapter_fingerprints")
    if not isinstance(fps, Mapping) or not fps or any(not isinstance(v, str) or not v for v in fps.values()):
        raise BindingRefusal("the body records no residual_adapter_fingerprints -- the served adapters' payload identity is "
                             "required to bind the record to what was served")
    layer = facts.get("expected_served_layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise BindingRefusal(f"the body's expected_served_layer={layer!r} is not an integer")
    if not _is_sha256_hex(object_sha256):
        raise BindingRefusal(f"object_sha256={object_sha256!r} is not a 64-hex sha256 -- the pins' object sha for this rung is required")
    if not _is_sha256_hex(served_object_sha256):
        raise BindingRefusal(f"served_object_sha256={served_object_sha256!r} is not a 64-hex sha256 -- the sha the bank view observed on "
                             "the bytes it bound to this rung's pin is required")
    if served_object_sha256 != object_sha256:
        raise BindingRefusal(f"served_object_sha256 {served_object_sha256} != object_sha256 {object_sha256} -- the bytes the view served "
                             "are not the requested rung's pinned bytes; a record naming bytes other than the ones served gets no identity")
    if not cell or not slurm_job_id or not serve_dir:
        raise BindingRefusal(f"cell={cell!r} / slurm_job_id={slurm_job_id!r} / serve_dir={serve_dir!r}: every identity field is required")
    return {"slurm_job_id": str(slurm_job_id), "serve_dir": str(serve_dir), "cell": str(cell),
            "residual_fingerprint": {str(k): v for k, v in sorted(fps.items())}, "object_sha256": object_sha256,
            "served_object_sha256": served_object_sha256, "expected_served_layer": int(layer)}


def stamp(body: Mapping[str, Any], **identity_kw: Any) -> Dict[str, Any]:
    """The persisted record: the body plus its `identity` (identity_for) plus `identity.evidence_sha256`."""
    rec = dict(body)
    rec[IDENTITY_KEY] = identity_for(body, **identity_kw)
    rec[IDENTITY_KEY][SHA_FIELD] = canonical_sha256(rec)
    return rec


def load_record(path: os.PathLike | str, *, expect_sha256: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """The record at `path`, re-hashed: returns (record, sha). Refuses by name an absent file, non-JSON, a record without
    the body's shape or the identity block, a missing identity field, a recorded sha that is not the re-hash, and (when
    `expect_sha256` is given -- the store's pointer) a sha other than the expected one."""
    p = Path(path)
    if not p.is_file():
        raise BindingRefusal(f"no serving-evidence record at {p}")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise BindingRefusal(f"{p} is not JSON: {e}") from None
    if not isinstance(rec, dict) or not isinstance(rec.get("facts"), dict) or not isinstance(rec.get("problems"), list):
        raise BindingRefusal(f"{p} is not the evidence body's record (facts + problems)")
    ident = rec.get(IDENTITY_KEY)
    if not isinstance(ident, dict):
        raise BindingRefusal(f"{p} carries no `{IDENTITY_KEY}` block -- an unidentified record cannot be bound to a store "
                             "(written before round 3, or by hand)")
    missing = [k for k in (*IDENTITY_FIELDS, SHA_FIELD) if k not in ident]
    if missing:
        raise BindingRefusal(f"{p}: identity lacks {missing}")
    recorded = ident[SHA_FIELD]
    got = canonical_sha256(rec)
    if recorded != got:
        raise BindingRefusal(f"{p}: identity.{SHA_FIELD} {str(recorded)[:12]}… != the record's re-hash {got[:12]}… -- the record "
                             "was edited after it was stamped, or is not a stamped record")
    if expect_sha256 is not None and got != expect_sha256:
        raise BindingRefusal(f"{p} hashes to {got[:12]}… but the store points at {str(expect_sha256)[:12]}… -- this is not the "
                             "record that certified the standup these rows were drawn under (another rung's, another job's, "
                             "or a retry's)")
    return rec, got


def fold_into(evidence: Optional[Dict[str, Any]], *, env: Optional[Mapping[str, str]] = None,
              log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Fold the launcher-named record's pointer into the served target's `evidence` (served_target.json carries it):
    `serving_evidence_sha256` + `serving_evidence_path`. A no-op when `ENV` is unset or empty, so an untouched evidence
    dict stays byte-identical. A named record that fails `load_record` REFUSES (SystemExit) -- before any generation,
    with the store untouched. Returns the sha when folded, else None."""
    environ = os.environ if env is None else env
    path = (environ.get(ENV) or "").strip()
    if not path or evidence is None:
        return None
    _rec, sha = load_record(path)
    evidence[EVIDENCE_SHA_KEY] = sha
    evidence[EVIDENCE_PATH_KEY] = str(path)
    if log:
        log(f"serving evidence bound to this store: {path} (sha256 {sha[:12]}…) -- the grader follows this pointer")
    return sha


def pointer_of(manifest: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """(sha, path) off ONE served_target.json, or None when the manifest carries no pointer."""
    ev = manifest.get("evidence") if isinstance(manifest, Mapping) else None
    if not isinstance(ev, Mapping) or EVIDENCE_SHA_KEY not in ev:
        return None
    return str(ev.get(EVIDENCE_SHA_KEY)), str(ev.get(EVIDENCE_PATH_KEY) or "")
