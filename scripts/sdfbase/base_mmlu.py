#!/usr/bin/env python
"""Base-model capability leg: PLAIN-TEXT 5-shot MMLU letter-logprob.

The existing letter-logprob path (`sisteer.eval.capability_ext.
_letter_logprob_accuracy`) is CHAT-shaped — it renders through
build_chat_prompt, which is wrong for a base model and doubly wrong for a
campaign whose whole data path bypasses chat templating. This script keeps
that path's load-bearing discipline by IMPORT — `_letter_token_ids`'s
single-token assertion, the per-question `obj.apply` context (the
skip_bos position-tracker lesson), `encode_kwargs`'s double-BOS guard —
and renders plain text: dev-split exemplars in the
`Question / A.–D. / Answer: X` convention, never build_chat_prompt.

Cells: `none` (bare base) plus any `<data-cell>__<arm>` whose fitted
object sits under --defenses-root/cells_all/<cell_id>/ (the campaign
banks — the live bank, or a snapshot bank such as /vol/out/<key>_r1 for
the round-1 winner objects), plus any cell named by an explicit
``--object <cell>=<path>`` (the round-2 DERIVED objects — V, P_resid, the
LoRA subtractions — live under derived/<kind>/<key>/<id>/, not in a bank's
cells_all/; the derived Modal app passes them this way). One model load;
objects swap per cell. n=500 test questions, seed 0, cais/mmlu —
accuracy + delta vs bare base with a binomial (Wilson) CI, reported not
gated.

Every artifact passes the shared identity gate BEFORE the model loads it,
and a DERIVED artifact additionally REQUIRES ``--parents-root`` (the fits
bank root): its parents' object and sidecar shas are re-verified against
the LIVE bank, so a refit parent under the same cell id can never be
paired with stale derived bytes.

Every run is NAMESPACED and PROVENANCED:

  * ``--run-label`` is REQUIRED and every output carries it in BOTH the
    directory and the filename — ``<out-dir>/<label>/mmlu_<label>_<cell>
    .json`` — so a snapshot-bank run (``r1``) and a refit run (``r2``) can
    never share a path or overwrite each other.
  * a shared RUN IDENTITY rides in every file: run label, model +
    revision, object sha256 (per cell; null for ``none``), the cais/mmlu
    revision actually loaded, seed / n / shots / scale, and the hash of the
    ORDERED question ids — the pairing key. Two rows files pair for
    McNemar only if their identities agree on dataset revision and qid
    hash; ``read_per_question`` hands both back so a consumer can assert
    that before computing anything.
  * per-cell aggregate + per-question rows are published TOGETHER,
    each via tmp+rename (rows first, then the aggregate), so no aggregate
    can exist without its rows and no half-written file is ever read;
    the summary is written only after the FULL requested set scored.

Per-question correctness is RETAINED in every run: each cell's rows file
(a header line with the run identity, then one row per question: qidx,
qid, subject, picked, answer, correct) makes a paired McNemar table
between any two cells pure post-hoc arithmetic — the earlier harvest kept
aggregates only and the 2–3 pp deltas could not be paired-tested
(summary.md open question 4).

Run inside the sdfbase-fits Modal app (--mode mmlu --run-label r2) or
anywhere with the repo env:

  PYTHONPATH=src python scripts/sdfbase/base_mmlu.py \
      --config configs/sdfbase_<key>_fits.yaml \
      --cells none,constv2-mix-1m__psr_ce_all \
      --defenses-root /vol/out/<key> --out-dir /vol/out/<key>/mmlu \
      --run-label r2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: 5-shot: the cais/mmlu dev split carries exactly five exemplars per
#: subject — the standard few-shot protocol.
DEFAULT_SHOTS = 5
LETTERS = ["A", "B", "C", "D"]
DATASET = "cais/mmlu"
DATASET_SPLIT = "test"

#: Run labels are path components AND filename stems: keep them boring.
_RUN_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
#: The hub dataset revision `datasets` bakes into its arrow cache path
#: (``.../cais___mmlu/all/0.0.0/<40-hex>/mmlu-test.arrow``) — the one
#: place the RESOLVED revision survives an offline load.
_CACHE_REV_RE = re.compile(r"/([0-9a-f]{40})/[^/]*\.arrow$")
_URL_REV_RE = re.compile(r"@([0-9a-f]{40})")


def render_question(question: str, choices: list[str],
                    answer_letter: str | None = None) -> str:
    """One question block, the plan-§11 plain-text convention. The
    exemplar form ends `Answer: X`; the scored form ends `Answer:` so the
    next token is the ` X` letter (matching `_letter_token_ids`'s leading-
    space encoding)."""
    lines = [f"Question: {question.strip()}"]
    lines += [f"{letter}. {str(c).strip()}"
              for letter, c in zip(LETTERS, choices)]
    lines.append(f"Answer: {answer_letter}" if answer_letter is not None
                 else "Answer:")
    return "\n".join(lines)


def render_prompt(row: dict, dev_rows: list[dict], shots: int) -> str:
    """Header + `shots` same-subject dev exemplars + the scored question.
    PLAIN TEXT ONLY — no chat template, ever (base models; the campaign's
    standing rule)."""
    subject = str(row.get("subject") or "").replace("_", " ").strip()
    about = f" about {subject}" if subject else ""
    parts = [f"The following are multiple choice questions (with answers)"
             f"{about}."]
    for ex in dev_rows[:shots]:
        parts.append(render_question(ex["question"], ex["choices"],
                                     LETTERS[int(ex["answer"])]))
    parts.append(render_question(row["question"], row["choices"]))
    return "\n\n".join(parts)


def dataset_revision_of(ds) -> str | None:
    """The hub revision a loaded `datasets` split actually came from, read
    off the artifacts the load left behind (no network): the 40-hex
    segment of the arrow cache path, else an ``@<sha>`` in a download
    checksum URL. None when neither carries one."""
    for cf in getattr(ds, "cache_files", None) or []:
        m = _CACHE_REV_RE.search(str(cf.get("filename", "")))
        if m:
            return m.group(1)
    info = getattr(ds, "info", None)
    for url in (getattr(info, "download_checksums", None) or {}):
        m = _URL_REV_RE.search(str(url))
        if m:
            return m.group(1)
    return None


def load_rows(n: int, seed: int, revision: str | None = None
              ) -> tuple[list[dict], dict[str, list[dict]], str]:
    """(test rows, dev exemplars by subject, resolved dataset revision)
    from cais/mmlu 'all'. The revision is part of every output's run
    identity: an explicit ``revision`` pins the load; the resolved one is
    read back off the loaded split and must agree with the pin. A load
    whose revision cannot be established at all REFUSES — an unpinned
    dataset cannot pair with anything."""
    from datasets import load_dataset

    kw = {"revision": revision} if revision else {}
    test = load_dataset(DATASET, "all", split=DATASET_SPLIT, **kw)
    resolved = dataset_revision_of(test)
    if resolved is None and revision is None:
        raise SystemExit(
            f"[base-mmlu] cannot establish the {DATASET} revision this load "
            "resolved to (no revision in the cache path or checksum URLs) "
            "— pass --dataset-revision <commit sha> so the run identity is "
            "pinned; McNemar pairing across runs requires it")
    if resolved is not None and revision is not None and resolved != revision:
        raise SystemExit(
            f"[base-mmlu] --dataset-revision {revision} but the loaded "
            f"split resolved to {resolved} — refusing to stamp a revision "
            "the data did not come from")
    test = test.shuffle(seed=seed).select(range(min(n, len(test))))
    rows = [{"question": r["question"], "choices": list(r["choices"]),
             "answer": int(r["answer"]), "subject": r["subject"]}
            for r in test]
    dev = load_dataset(DATASET, "all", split="dev", **kw)
    dev_by_subject: dict[str, list[dict]] = {}
    for r in dev:
        dev_by_subject.setdefault(r["subject"], []).append(
            {"question": r["question"], "choices": list(r["choices"]),
             "answer": int(r["answer"])})
    missing = sorted({r["subject"] for r in rows} - set(dev_by_subject))
    if missing:
        raise SystemExit(f"[base-mmlu] test subjects with no dev exemplars "
                         f"(broken few-shot protocol): {missing}")
    return rows, dev_by_subject, str(resolved or revision)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def question_id(row: dict) -> str:
    """A content-derived stable id (sha256-16 of subject + question text),
    so two runs pair rows even if the draw order ever changes — qidx alone
    is only stable under the same (n, seed)."""
    return hashlib.sha256(
        f"{row['subject']}\x00{row['question']}".encode()).hexdigest()[:16]


def ordered_qid_hash(rows: list[dict]) -> str:
    """sha256 over the ORDERED question ids — the pairing key two runs must
    share before any McNemar table is computed (same draw, same order)."""
    return hashlib.sha256(
        "\n".join(question_id(r) for r in rows).encode()).hexdigest()


def score_cell(model, tok, obj, rows: list[dict],
               dev_by_subject: dict, shots: int, scale: float = 1.0
               ) -> tuple[dict, list[dict]]:
    """(aggregate, per-question rows) for one cell. Discipline imported
    from capability_ext: per-question `obj.apply` context (never one
    long-lived context — the skip_bos tracker counts per forward),
    `_letter_token_ids`'s single-token assertion, `encode_kwargs`'s
    double-BOS guard. Rendering is plain text (this module), never
    build_chat_prompt.

    Each per-question row: qidx (draw order), qid (content hash), subject,
    picked letter, answer letter, correct — the paired win/loss table
    between ANY two cells is derivable from two rows files (module
    docstring). The aggregate is DERIVED from the rows, so the two can
    never disagree."""
    import torch

    from sisteer.eval.capability_ext import (_letter_token_ids,
                                             _maybe_inject, _obj_ctx)
    from sisteer.generate import encode_kwargs

    device = next(model.parameters()).device
    letter_ids = _letter_token_ids(tok, LETTERS)
    per_q: list[dict] = []
    with torch.no_grad():
        for qidx, row in enumerate(rows):
            prompt = render_prompt(row, dev_by_subject[row["subject"]],
                                   shots)
            prompt = _maybe_inject(obj, prompt)
            enc = tok(prompt, return_tensors="pt",
                      **encode_kwargs(tok, prompt)).to(device)
            with _obj_ctx(model, obj, scale):
                logits = model(**enc).logits[0, -1]
            pick = int(logits[letter_ids].argmax())
            per_q.append({
                "qidx": qidx,
                "qid": question_id(row),
                "subject": row["subject"],
                "picked": LETTERS[pick],
                "answer": LETTERS[int(row["answer"])],
                "correct": bool(pick == row["answer"]),
            })
    n = len(rows)
    correct = sum(r["correct"] for r in per_q)
    lo, hi = wilson_ci(correct, n)
    agg = {"acc": correct / max(n, 1), "correct": correct, "n": n,
           "ci95": [lo, hi]}
    return agg, per_q


# --------------------------------------------------------------------------
# Namespaced, provenanced, atomic publication
# --------------------------------------------------------------------------

PER_QUESTION_HEADER_KIND = "mmlu_per_question_header"


def validate_run_label(label: str) -> str:
    if not label or not _RUN_LABEL_RE.match(label):
        raise SystemExit(
            f"[base-mmlu] --run-label {label!r} must be a plain path/"
            "filename token ([A-Za-z0-9][A-Za-z0-9_.-]*) — it names the "
            "output dir AND every filename")
    return label


def run_dir_for(out_dir: Path, run_label: str) -> Path:
    """``<out-dir>/<label>/`` — every file of a run lives under its label."""
    return Path(out_dir) / validate_run_label(run_label)


def cell_paths(run_dir: Path, run_label: str, cell: str
               ) -> tuple[Path, Path]:
    """(aggregate json, per-question jsonl) for one cell — label in the
    filename too, so two runs' files can never collide even if a consumer
    flattens directories."""
    stem = f"mmlu_{run_label}_{cell}"
    return run_dir / f"{stem}.json", run_dir / f"{stem}.per_question.jsonl"


def summary_path(run_dir: Path, run_label: str) -> Path:
    return run_dir / f"mmlu_{run_label}_summary.json"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_cell_results(run_dir: Path, run_label: str, cell: str,
                       agg: dict, per_q: list[dict], identity: dict
                       ) -> tuple[Path, Path]:
    """Publish one cell: the per-question rows file FIRST (header line
    carrying the run identity, then one JSON row per question in draw
    order), then the aggregate (the same identity block + the stats),
    each through tmp+rename — a reader sees either nothing or a whole
    file, and never an aggregate whose rows are missing. Returns
    (aggregate path, rows path)."""
    agg_path, pq_path = cell_paths(run_dir, run_label, cell)
    if sum(r["correct"] for r in per_q) != agg["correct"] \
            or len(per_q) != agg["n"]:
        raise SystemExit("[base-mmlu] internal: per-question rows disagree "
                         "with the aggregate — refusing to publish either")
    header = {"kind": PER_QUESTION_HEADER_KIND, "cell": cell,
              "n_rows": len(per_q), "identity": identity}
    _atomic_write(pq_path, "".join(
        json.dumps(r) + "\n" for r in [header, *per_q]))
    record = {**agg, "cell": cell, "identity": identity,
              "per_question_file": pq_path.name}
    _atomic_write(agg_path, json.dumps(record, indent=1) + "\n")
    return agg_path, pq_path


def read_per_question(path: Path) -> tuple[dict, list[dict]]:
    """(run identity, rows) from a per-question file — the ONE reader.
    Refuses a file without the header or whose row count disagrees with
    it (a torn write, or a pre-identity file), so a McNemar consumer can
    assert identity agreement (dataset revision, ordered-qid hash) before
    computing anything."""
    lines = [json.loads(ln) for ln in
             Path(path).read_text(encoding="utf-8").splitlines() if ln]
    if not lines or lines[0].get("kind") != PER_QUESTION_HEADER_KIND:
        raise SystemExit(f"[base-mmlu] {path} carries no run-identity "
                         "header — not a pairable per-question file")
    header, rows = lines[0], lines[1:]
    if len(rows) != int(header.get("n_rows", -1)):
        raise SystemExit(f"[base-mmlu] {path} header promises "
                         f"{header.get('n_rows')} rows, file has "
                         f"{len(rows)} — torn write")
    return header["identity"], rows


def assert_pairable(identity_a: dict, identity_b: dict) -> None:
    """Two runs pair only on the same draw of the same data: equal
    dataset revision, ordered-qid hash, n and shots. Named refusal."""
    for k in ("dataset", "dataset_revision", "ordered_qid_hash", "n",
              "shots"):
        if identity_a.get(k) != identity_b.get(k):
            raise SystemExit(
                f"[base-mmlu] runs are not pairable: {k} differs "
                f"({identity_a.get(k)!r} vs {identity_b.get(k)!r})")


def resolve_object_path(defenses_root, cell: str, *, parents_root=None,
                        explicit_path=None, consumer: str = "base-mmlu"
                        ) -> Path:
    """The artifact for ``cell``: ``<defenses_root>/cells_all/<cell>/
    <artifact>`` — the campaign banks' layout (positions=all leg) — or,
    with ``explicit_path`` (an ``--object <cell>=<path>`` entry: a derived
    object under derived/<kind>/<key>/<id>/), that file. Artifact name via
    the registry's own convention; load_object dispatches on file CONTENT,
    so a missing name falls back to whichever object.* exists in the dir
    (sidecars such as object.pt.provenance.json are not artifacts).

    The resolved artifact passes the shared identity gate
    (``train_budget.assert_object_identity``) BEFORE anything loads it:
    a fitted cell's sidecar must name THIS cell, the object bytes must
    match the sidecar's sha, the budget kind must validate, and a present
    COMPLETE manifest must be coherent; a derived object's sidecar must
    bind the bytes to verified parent identities AND — ``parents_root``
    being REQUIRED for a derived object — its parents are re-verified by
    sha against the live bank; a wrong-cell, stale or unprovable artifact
    refuses in milliseconds, never after the model load. Resume-grade
    strictness on the sample manifest (a stated-absence null passes; this
    is a measurement consumer, not the two_si serving gate)."""
    from sisteer.eval.object_identity import (assert_object_identity,
                                              assert_parents_root_for_derived)
    from sisteer.methods import object_artifact_name

    if explicit_path is not None:
        p = Path(explicit_path)
        if not p.is_file():
            raise SystemExit(f"[{consumer}] --object {cell}={p}: no such file")
    else:
        if "__" not in cell:
            raise SystemExit(f"[{consumer}] cell {cell!r} is not "
                             "<data-cell>__<arm> (or 'none')")
        if defenses_root is None:
            raise SystemExit(f"[{consumer}] cell {cell!r} needs --defenses-root "
                             f"(or an explicit --object {cell}=<path>)")
        arm = cell.split("__", 1)[1]
        cell_dir = Path(defenses_root) / "cells_all" / cell
        p = cell_dir / object_artifact_name(arm)
        if not p.exists():
            # sidecars (object.pt.provenance.json) are not artifacts
            candidates = sorted(q for q in cell_dir.glob("object.*")
                                if q.suffix != ".json")
            if len(candidates) != 1:
                raise SystemExit(
                    f"[{consumer}] no unambiguous artifact for {cell!r} under "
                    f"{cell_dir} (found {[c.name for c in candidates]}) — is "
                    "the cell COMPLETE and staged?")
            p = candidates[0]
    assert_parents_root_for_derived(p, parents_root, consumer=consumer)
    assert_object_identity(p, cell, consumer=consumer,
                           parents_root=parents_root)
    return p


#: The run-identity keys two invocations under ONE label must agree on for
#: their summaries to merge: a label is one draw of one dataset revision
#: under one model, one prompt shape and one scale. (defenses_root /
#: parents_root / config path are per-cell provenance, carried by the cell
#: files, and may legitimately differ between invocations.)
SUMMARY_MERGE_KEYS = ("run_label", "model", "model_revision", "dataset",
                      "dataset_split", "dataset_revision", "n", "seed",
                      "shots", "scale", "ordered_qid_hash", "render",
                      # runtime identity:
                      # the load dtype and attention backend change the
                      # logits a letter argmax / greedy pick is taken from
                      "dtype", "attn_implementation")

#: Identity keys DERIVED FROM THE FITS CONFIG. An identity written before
#: they were stamped (the wave-A r1/r2/r3 summaries) lacks them; it is
#: accepted iff its ``config_sha256`` equals the new run's — the settings
#: come from that config, so equal bytes prove equal settings. Any other
#: absent key is unverifiable and refuses.
CONFIG_DERIVED_KEYS = ("dtype", "attn_implementation")


def _identity_mismatch(prior: dict, new: dict, keys) -> str | None:
    for k in keys:
        if k not in prior:
            if k in CONFIG_DERIVED_KEYS and prior.get("config_sha256") \
                    and prior.get("config_sha256") == new.get("config_sha256"):
                continue  # legacy identity; equal config bytes prove it
            return (f"{k} is absent from the existing identity and cannot "
                    f"be verified (config_sha256 {prior.get('config_sha256')!r}"
                    f" vs {new.get('config_sha256')!r})")
        if prior.get(k) != new.get(k):
            return f"{k} differs ({prior.get(k)!r} vs {new.get(k)!r})"
    return None


def preflight_label(run_dir: Path, run_label: str, run_identity: dict,
                    cell_shas: dict, *, cell_paths_fn, summary_path_fn,
                    merge_keys=SUMMARY_MERGE_KEYS,
                    consumer: str = "base-mmlu") -> list[str]:
    """Refuse BEFORE the model loads or anything is written if this
    invocation would contaminate an existing label: the identity check
    used to run only at the summary merge, after every per-cell file had
    already landed under the label.

    ``cell_shas`` = {cell: object_sha256 | None (bare)} for every cell this
    invocation will publish, resolved through the identity gate first (no
    GPU). Refusals, all named:
      * an existing summary of the label whose identity differs on any of
        ``merge_keys`` — a label is ONE run identity;
      * an existing per-cell aggregate whose identity differs on any of
        ``merge_keys`` — same rule, at the file this run would overwrite;
      * an existing per-cell aggregate scored on a DIFFERENT object
        (``identity.object_sha256``) — a label holds one object per cell,
        and overwriting it would destroy the record another generation of
        that cell was published under (use a new label instead).
    An existing per-cell file with the same identity AND object sha is an
    idempotent re-score and is allowed (returned, so the caller can say so).
    """
    run_dir = Path(run_dir)
    spath = summary_path_fn(run_dir, run_label)
    if spath.exists():
        prior = (json.loads(spath.read_text(encoding="utf-8"))
                 .get("run") or {})
        why = _identity_mismatch(prior, run_identity, merge_keys)
        if why:
            raise SystemExit(
                f"[{consumer}] label {run_label!r} already holds a summary "
                f"({spath}) whose {why} — a label is ONE run identity; "
                "refusing before any output is written (use a new label)")
    rescored: list[str] = []
    for cell, sha in cell_shas.items():
        agg_path, _pq = cell_paths_fn(run_dir, run_label, cell)
        if not agg_path.exists():
            continue
        prior = (json.loads(agg_path.read_text(encoding="utf-8"))
                 .get("identity") or {})
        why = _identity_mismatch(prior, run_identity, merge_keys)
        if why:
            raise SystemExit(
                f"[{consumer}] label {run_label!r} already holds {cell!r} "
                f"({agg_path.name}) whose {why} — refusing to overwrite it "
                "(a label is ONE run identity; use a new label)")
        if prior.get("object_sha256") != sha:
            raise SystemExit(
                f"[{consumer}] label {run_label!r} already holds {cell!r} "
                f"scored on object {prior.get('object_sha256')!r}; this run "
                f"would score {sha!r} — a label holds ONE object per cell, "
                "and overwriting would destroy that record (use a new "
                "label for the new object generation)")
        rescored.append(cell)
    return rescored


def write_summary(spath: Path, run_identity: dict, requested_cells: list,
                  results: dict, *, consumer: str = "base-mmlu",
                  merge_keys=SUMMARY_MERGE_KEYS) -> dict:
    """Publish the run summary, MERGING into an existing summary of the same
    label (2026-09-03): a label is a NAMESPACE of cells scored under one run
    identity, and a later invocation that adds cells to it (the 2x-dose
    lane topping the ``r2`` label up with the cells the wave-A run did not
    score) must not erase the record of the earlier one. The merged
    summary carries the union of ``requested_cells`` (first-seen order),
    every cell's result (a re-scored cell replaces its own entry only), and
    an ``invocations`` list — one entry per publishing run. An existing
    summary whose identity differs on any of ``merge_keys`` is REFUSED: two
    identities under one label is exactly the collision the label exists
    to prevent. Each invocation still writes only after ITS full requested
    set scored (the caller's contract), and the write is tmp+rename."""
    from datetime import datetime, timezone

    prior = None
    if Path(spath).exists():
        prior = json.loads(Path(spath).read_text(encoding="utf-8"))
        why = _identity_mismatch(prior.get("run") or {}, run_identity,
                                 merge_keys)
        if why:
            raise SystemExit(
                f"[{consumer}] {spath} already holds a summary for label "
                f"{run_identity.get('run_label')!r} whose {why} — a label is "
                "ONE run identity; refusing to merge or overwrite")
    cells = dict((prior or {}).get("cells") or {})
    cells.update(results)
    requested = list((prior or {}).get("requested_cells") or [])
    requested += [c for c in requested_cells if c not in requested]
    invocations = list((prior or {}).get("invocations") or [])
    if prior is not None and not invocations:
        # a pre-merge summary: record it as the first invocation
        invocations.append({"written_at": None,
                            "requested_cells": list(
                                prior.get("requested_cells") or []),
                            "note": "summary written before invocation "
                                    "tracking"})
    invocations.append({
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_cells": list(requested_cells)})
    summary = {"run": run_identity, "requested_cells": requested,
               "cells": cells, "invocations": invocations}
    _atomic_write(Path(spath), json.dumps(summary, indent=1) + "\n")
    return summary


def parse_explicit_objects(entries, cells: list[str]) -> dict[str, str]:
    """``--object CELL=PATH`` entries -> {cell: path}; every named cell
    must be in ``cells`` (an object for a cell that is not scored states
    two intents) and a cell may be named once."""
    explicit: dict[str, str] = {}
    for kv in entries or []:
        if "=" not in kv:
            raise SystemExit(f"[base-mmlu] --object {kv!r} is not CELL=PATH")
        c, pth = kv.split("=", 1)
        if c in explicit:
            raise SystemExit(f"[base-mmlu] --object names {c!r} twice")
        explicit[c] = pth
    unknown = sorted(set(explicit) - set(cells))
    if unknown:
        raise SystemExit(f"[base-mmlu] --object names cells not in --cells: "
                         f"{unknown}")
    return explicit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="the campaign fits config (model/revision pins)")
    ap.add_argument("--cells", required=True,
                    help="comma list: 'none' and/or <data-cell>__<arm> ids "
                         "staged under --defenses-root/cells_all/")
    ap.add_argument("--defenses-root", default=None,
                    help="bank root holding cells_all/<cell_id>/ (required "
                         "for every fitted cell not named by --object); the "
                         "live bank or a snapshot bank (<key>_r1)")
    ap.add_argument("--object", action="append", default=[],
                    metavar="CELL=PATH",
                    help="explicit artifact path for a cell in --cells (the "
                         "DERIVED objects: derived/<kind>/<key>/<id>/object.pt "
                         "— derived ids are not bank cells); passes the same "
                         "identity gate and REQUIRES --parents-root")
    ap.add_argument("--parents-root", default=None,
                    help="fits bank root (<key>/cells_all/<cell>/) against "
                         "which a DERIVED object's parents are re-verified "
                         "by sha — REQUIRED for a derived object (a refit "
                         "parent under the same cell id must never be paired "
                         "with stale derived bytes)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dataset-revision", default=None,
                    help=f"pin the {DATASET} hub revision (commit sha); "
                         "otherwise the revision the load resolved to is "
                         "recorded, and a load that cannot establish one "
                         "refuses")
    ap.add_argument("--out-dir", required=True,
                    help="output ROOT: files land under <out-dir>/<run-"
                         "label>/mmlu_<run-label>_<cell>{.json,"
                         ".per_question.jsonl} + the run summary")
    ap.add_argument("--run-label", required=True,
                    help="REQUIRED run namespace (e.g. r1 for the snapshot "
                         "bank, r2 for the refit bank): dir + filename "
                         "component, so two runs never share a path")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    import yaml

    from sisteer.eval.defense_cell import sha256_file
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.methods import load_object
    from sisteer.models import load_model_and_tokenizer

    run_label = validate_run_label(args.run_label)
    pin_attention_backends()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    cells = [c for c in args.cells.split(",") if c]
    if len(set(cells)) != len(cells):
        raise SystemExit(f"[base-mmlu] duplicate cells in --cells: {cells}")
    explicit = parse_explicit_objects(args.object, cells)
    if any(c != "none" and c not in explicit for c in cells) \
            and not args.defenses_root:
        raise SystemExit("[base-mmlu] fitted cells given but no "
                         "--defenses-root (or --object CELL=PATH for each)")
    # Bare base FIRST (stable otherwise): every fitted cell's aggregate
    # then carries delta_vs_bare at publish time, from THIS run's bare row.
    cells = sorted(cells, key=lambda c: c != "none")

    # Identity gate + object shas for EVERY cell first (milliseconds, no
    # GPU), then the label preflight, then the data and the model: a
    # wrong-cell artifact or a label collision refuses before anything is
    # loaded or written.
    resolved: dict[str, Path | None] = {}
    for cell in cells:
        resolved[cell] = None if cell == "none" else resolve_object_path(
            Path(args.defenses_root) if args.defenses_root else None,
            cell, parents_root=args.parents_root,
            explicit_path=explicit.get(cell))
    cell_shas = {c: (sha256_file(p) if p else None)
                 for c, p in resolved.items()}

    rows, dev_by_subject, dataset_revision = load_rows(
        args.n, args.seed, revision=args.dataset_revision)

    run_identity = {
        "run_label": run_label,
        "model": cfg["model"], "model_revision": cfg.get("model_revision"),
        "config": str(args.config),
        "config_sha256": sha256_file(cfg_path),
        "dataset": DATASET, "dataset_split": DATASET_SPLIT,
        "dataset_revision": dataset_revision,
        "n": len(rows), "requested_n": args.n, "seed": args.seed,
        "shots": args.shots, "scale": float(args.scale),
        "ordered_qid_hash": ordered_qid_hash(rows),
        "render": "plain_text_5shot", "chat_template_bypassed": True,
        "dtype": cfg.get("dtype", "bfloat16"),
        "attn_implementation": cfg.get("attn_implementation", "sdpa"),
        "defenses_root": (str(args.defenses_root)
                          if args.defenses_root else None),
        "parents_root": (str(args.parents_root)
                         if args.parents_root else None),
        "explicit_objects": explicit or None,
    }
    run_dir = run_dir_for(Path(args.out_dir), run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    rescored = preflight_label(run_dir, run_label, run_identity, cell_shas,
                               cell_paths_fn=cell_paths,
                               summary_path_fn=summary_path)
    if rescored:
        print(f"[base-mmlu] {run_label}: re-scoring {rescored} (same "
              "identity + object sha already published; idempotent)")

    model, tok = load_model_and_tokenizer(
        cfg["model"], dtype=cfg.get("dtype", "bfloat16"),
        revision=cfg.get("model_revision"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()

    results: dict[str, dict] = {}
    base_acc = None
    for cell in cells:
        path = resolved[cell]
        obj = load_object(path, model=model, tok=tok) if path else None
        res, per_q = score_cell(model, tok, obj, rows, dev_by_subject,
                                args.shots, scale=args.scale)
        res["delta_vs_bare"] = (round(res["acc"] - base_acc, 6)
                                if base_acc is not None and cell != "none"
                                else None)
        identity = {**run_identity, "cell": cell,
                    "object_path": str(path) if path else None,
                    "object_sha256": cell_shas[cell]}
        agg_path, pq_path = write_cell_results(run_dir, run_label, cell,
                                               res, per_q, identity)
        results[cell] = {**res, "identity": identity,
                         "per_question_file": pq_path.name,
                         "aggregate_file": agg_path.name}
        if cell == "none":
            base_acc = res["acc"]
        print(f"[base-mmlu] {run_label}/{cell}: acc={res['acc']:.4f} "
              f"({res['correct']}/{res['n']}, CI95 "
              f"[{res['ci95'][0]:.4f}, {res['ci95'][1]:.4f}]) "
              f"-> {agg_path.name} + {pq_path.name}")

    # The summary exists only once the FULL requested set scored — a
    # partial run leaves per-cell pairs and no summary, never a summary
    # that claims cells it did not score. Merged into an existing summary
    # of the same label + identity (write_summary), never overwriting it.
    spath = summary_path(run_dir, run_label)
    write_summary(spath, run_identity, cells, results)
    print(f"[base-mmlu] wrote {spath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
