#!/usr/bin/env python
"""The GENERATION-ONLY sidecar proof of a misalign attempt (fj29): over a pulled attempt root, every model that served or is named by the
attempt's own records, and ZERO uses of every REMOVED (and, reported, every ADDED) Petri chain leg.

A misalign row is Petri-MODE (`seedrep_launch_plan.leg_table(...)["misalign"]["mode"] == "petri"`), so a producer difference in
`petri_runner.py` (the archived -> launch judge fallback chain, dbf326ead; the re-key guard, f29562721) requires the covering sidecar
proof (fj28). The misalign attempt's store carries NO Petri completion bundle: its children run `sisteer.capeval.misalign.runner`
(generations through the proxy, single-turn agentic scenarios -- no auditor), and its JUDGING is a grading-host step downstream of the
pull (`scripts/sdfbase/full22/misalign/misalign_judge.py`, claude-sonnet-5), never part of the accepted attempt. So the proof the rule
asks for -- "the removed / added legs were never reached in this store" -- is read off the attempt's own records:

    exp/misalign/<family>__s<seed>__<behavior>/<cell>/meta.json        served_models, virtual_model, params (the runner's record)
    exp/misalign/<family>__s<seed>__<behavior>/<cell>/responses.jsonl  one row per (item, sample): model_reported, virtual_model, ...
    exp/misalign/<family>__s<seed>__<behavior>/response_evidence.jsonl the proxy's record of EVERY response it served: vmodel per request
    modal_evidence.json                                                the attempt's completion evidence (bound to the attempt)

Every JSON string value of every record is scanned for each leg's model id and label (case-insensitive substring); a hit is a USE. The
record is written in `seedrep_carry_over.petri_sidecar_proof`'s shape -- `ok`, `cells`, `removed_leg.uses` (+ per leg / per cell),
`added_legs`, `served_models`, `judge`, `inputs` = every file read by path + sha256, `why` -- so the certifying chain can hold it to the
volume bytes by digest (fj28's canonical-root + digest binding).

    misalign_generation_proof.py --root <pulled attempt root> --volume-root <the root's /vol/... spelling> --out <proof.json>
                                 --prior-dir <RS/prior/<sha7>/ (the archived run state)> --standup-id <the consuming row's id>
                                 --prev-w <the archived tree> --this-w <the launch tree> [--repo <git objects>] [--chain-note <text>]

The POLICY (fj31 / fj33): the removed / added chain identities and the target set are derived by `seedrep_carry_over.generation_proof_policy`
over the ARCHIVED ROW -- the same function, over the same row, the consumer (`generation_proof_record`) derives them with and holds the
document's `policy` block EQUAL to -- so a document this CLI writes is a document the certificate can admit; there is no second policy assembly
here (fj33: the CLI's own `chain_policy(...) + target` lacked the row's `cells`, so every document it wrote was refused).

Exit 0 whenever the proof was DERIVED (ok true or false -- the verdict is in the record); 2 when the root, the archive or a tree cannot be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Petri's removed auditor leg (`seedrep_carry_over.PETRI_REMOVED_LEG_MODELS` / `_LABELS`, spelled here so this script imports nothing
#: from scripts/modal) and the judge chain the archived -> launch petri_runner difference ADDED (dbf326ead: claude-opus-4-8, then grok-4.6)
DEFAULT_REMOVED_LEGS: Tuple[str, ...] = ("openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated", "qwen-abl")
DEFAULT_ADDED_LEGS: Tuple[str, ...] = ("anthropic/claude-opus-4-8", "openrouter/x-ai/grok-4.6")
SCHEMA = "seedrep_misalign_generation_proof/1"
JUDGE_SITE = ("the grading host, downstream of the pull: scripts/sdfbase/full22/misalign/misalign_judge.py (D8, claude-sonnet-5) writes "
              "RS/grades/misalign/<TAG>/.../judged.jsonl -- never part of the accepted attempt")


def _sha(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


def _strings(obj: Any) -> Iterable[str]:
    """Every string value (and key) in a JSON document."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def _hits(doc: Any, legs: Sequence[str]) -> Dict[str, int]:
    out = {leg: 0 for leg in legs}
    for s in _strings(doc):
        low = s.lower()
        for leg in legs:
            if leg.lower() in low:
                out[leg] += 1
    return out


def _add(a: Dict[str, int], b: Mapping[str, int]) -> None:
    for k, v in b.items():
        a[k] = a.get(k, 0) + int(v)


#: the manifest keys that spell the Petri chains, by ROLE (fj31: the removed / added model identities are DERIVED from the two trees'
#: manifests through git, per role, never taken from a document or a constant)
MANIFEST_CHAIN_KEYS: Tuple[Tuple[str, str], ...] = (("auditor", "auditor_model"), ("auditor", "auditor_fallback_models"), ("judge", "judge_model"), ("judge", "judge_fallback_models"))
#: the file KINDS a generation-only proof reads (its input set is DERIVED from the archived row's children, never taken from a document)
KIND_META, KIND_ROWS, KIND_EVIDENCE, KIND_MODAL_EVIDENCE = "meta", "rows", "response_evidence", "modal_evidence"
#: the verdict fields a recorded proof must reproduce field by field when re-derived
VERDICT_KEYS: Tuple[str, ...] = ("ok", "cells", "n_rows", "n_response_evidence", "removed_leg", "added_legs", "served_models", "models_reported", "per_cell", "inputs",
                                 "bytes_read", "target_models", "foreign_models", "policy")


def parse_manifest_chains(text: str) -> Dict[str, List[str]]:
    """The chain keys of a Petri manifest (`MANIFEST_CHAIN_KEYS`) from its YAML text, by a line reader for exactly that shape: a scalar
    `key: value  # comment` or a list `key:` followed by `- item  # comment` lines at a deeper indent. Every key found is returned (a scalar as a
    one-item list); keys absent are absent. Stdlib-only, so the certificate needs no YAML library on its path."""
    out: Dict[str, List[str]] = {}
    lines = text.splitlines()
    keys = {k for _r, k in MANIFEST_CHAIN_KEYS}
    n = 0
    while n < len(lines):
        raw = lines[n]
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.strip()
        m = None
        for k in keys:
            if stripped.startswith(k + ":"):
                m = k
                break
        if m is None:
            n += 1
            continue
        value = stripped[len(m) + 1:].strip()
        indent = len(line) - len(line.lstrip())
        if value:
            out[m] = [value.strip("'\"")]
            n += 1
            continue
        items: List[str] = []
        n += 1
        while n < len(lines):
            nxt = lines[n].split("#", 1)[0].rstrip()
            if not nxt.strip():
                n += 1
                continue
            ind = len(nxt) - len(nxt.lstrip())
            if ind <= indent or not nxt.strip().startswith("- "):
                break
            items.append(nxt.strip()[2:].strip().strip("'\""))
            n += 1
        out[m] = items
    return out


def chain_policy(git: Callable[[Sequence[str]], Tuple[int, str]], *, prev_W: str, this_W: str, manifests: Sequence[str],
                 label_of: Callable[[str], str]) -> Dict[str, Any]:
    """The REMOVED / ADDED chain-leg identities between the two trees, derived from ``manifests`` (the archived row's) read at both trees through
    ``git`` and diffed PER ROLE (auditor: auditor_model + auditor_fallback_models; judge: judge_model + judge_fallback_models), the union across
    roles; each identity carries its alias from ``label_of`` (the tree's `auditor_label`). A manifest unreadable at either tree, or naming no chain
    key, REFUSES (ValueError) -- there is no constant to fall back on."""
    if not manifests:
        raise ValueError("the archived row names no manifest -- the chain policy cannot be derived")
    prev_chains: Dict[str, Dict[str, List[str]]] = {}
    this_chains: Dict[str, Dict[str, List[str]]] = {}
    removed: set = set()
    added: set = set()
    for mf in sorted(set(manifests)):
        rc_a, a = git(["show", f"{prev_W}:{mf}"])
        rc_b, b = git(["show", f"{this_W}:{mf}"])
        if rc_a != 0 or rc_b != 0:
            raise ValueError(f"manifest {mf} is not readable at {'the prior tree ' + prev_W[:12] if rc_a != 0 else ''}{' and ' if rc_a != 0 and rc_b != 0 else ''}"
                             f"{'this tree ' + this_W[:12] if rc_b != 0 else ''} -- the chain policy cannot be derived")
        ca, cb = parse_manifest_chains(a), parse_manifest_chains(b)
        if not ca and not cb:
            raise ValueError(f"manifest {mf} names no chain key ({sorted(k for _r, k in MANIFEST_CHAIN_KEYS)}) at either tree -- the chain policy cannot be derived")
        prev_chains[mf], this_chains[mf] = ca, cb
        for role in ("auditor", "judge"):
            pa = {m for r_, k in MANIFEST_CHAIN_KEYS if r_ == role for m in ca.get(k, [])}
            pb = {m for r_, k in MANIFEST_CHAIN_KEYS if r_ == role for m in cb.get(k, [])}
            removed |= pa - pb
            added |= pb - pa
    def spell(models: set) -> List[str]:
        out: set = set()
        for m in models:
            out.add(m)
            lab = str(label_of(m)).strip()
            if lab:
                out.add(lab)
        return sorted(out)
    return {"prev_W": prev_W, "this_W": this_W, "manifests": sorted(set(manifests)), "prev": prev_chains, "this": this_chains,
            "removed_models": sorted(removed), "added_models": sorted(added), "removed": spell(removed), "added": spell(added)}


def _argv_value(argv: Sequence[Any], flag: str) -> Optional[str]:
    for i, a in enumerate(argv):
        a = str(a)
        if a == flag and i + 1 < len(argv):
            return str(argv[i + 1])
        if a.startswith(flag + "="):
            return a[len(flag) + 1:]
    return None


def expected_inputs(children: Sequence[Mapping[str, Any]], root: str) -> Dict[str, str]:
    """``{volume path: kind}`` of every file a generation-only proof MUST read for an attempt whose children are ``children`` (the ARCHIVED plan
    row's, the evaluator's `prev_row`): per child its `--out` store dir (`{attempt_root}` spelled as ``root``) and `--only-cells` cell ->
    `<out>/<cell>/meta.json` + `responses.jsonl`; per distinct store dir `response_evidence.jsonl`; the root's `modal_evidence.json`. A child
    naming no `--out` / `--only-cells` refuses -- the set is derived from the plan, never from a document."""
    root = str(root).rstrip("/")
    out: Dict[str, str] = {f"{root}/modal_evidence.json": KIND_MODAL_EVIDENCE}
    for k in children:
        argv = list(k.get("argv") or [])
        store, cell = _argv_value(argv, "--out"), _argv_value(argv, "--only-cells")
        if not store or not cell:
            raise ValueError(f"child {k.get('arm')!r} s{k.get('s')} names no --out / --only-cells -- its store cannot be located from the plan")
        store = store.replace("{attempt_root}", root).rstrip("/")
        if not store.startswith(root + "/"):
            raise ValueError(f"child {k.get('arm')!r}: store {store} is not under the attempt root {root}")
        out[f"{store}/response_evidence.jsonl"] = KIND_EVIDENCE
        out[f"{store}/{cell}/meta.json"] = KIND_META
        out[f"{store}/{cell}/responses.jsonl"] = KIND_ROWS
    return out


def derive_over(read_text: Callable[[str], Optional[str]], *, root: str, files: Mapping[str, str], removed: Sequence[str], added: Sequence[str],
                target: Sequence[str], policy: Optional[Mapping[str, Any]] = None, chain_note: Optional[str] = None) -> Dict[str, Any]:
    """The proof over ``files`` (`expected_inputs`' shape) read through ``read_text`` (a volume reader, a pulled-root reader, a test's dict): every
    JSON string of every record scanned for each REMOVED and each ADDED leg's identity (model id and alias, ``removed`` / ``added`` -- the
    tree-derived policy, never a document's list); every observed model name (`served_models`, `virtual_model`, `model_reported`, `vmodel`) held
    to the ``target`` set (the archived row's cells as this tree spells their virtual models); `inputs` = every path with the sha256 of the text
    read (utf-8); `bytes_read` the total. `ok` iff cells were read AND zero removed-leg uses AND zero added-leg uses AND no foreign name -- each
    failure named in `why`. An absent or unreadable input REFUSES (FileNotFoundError / ValueError): nothing is proven over a partial store."""
    root = str(root).rstrip("/")
    inputs: Dict[str, str] = {}
    per_cell: Dict[str, Dict[str, Any]] = {}
    removed_total = {leg: 0 for leg in removed}
    added_total = {leg: 0 for leg in added}
    served: set = set()
    reported: set = set()
    observed: set = set()
    n_rows = n_evidence = 0
    bytes_read = 0
    texts: Dict[str, str] = {}
    target_set = {str(t) for t in target}
    for path in sorted(files):
        if not str(path).startswith(root + "/"):
            raise ValueError(f"{path}: not under the attempt root {root}")
        text = read_text(path)
        if text is None:
            raise FileNotFoundError(f"{path}: absent -- the store is incomplete, nothing is proven")
        data = text.encode("utf-8")
        inputs[path] = _sha(data)
        bytes_read += len(data)
        texts[path] = text
    bundle: Any = "modal_evidence.json not read"
    for path, kind in sorted(files.items()):
        text = texts[path]
        if kind == KIND_EVIDENCE:
            for line in text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_evidence += 1
                if rec.get("vmodel") is not None:
                    served.add(str(rec["vmodel"])); observed.add(str(rec["vmodel"]))
                _add(removed_total, _hits(rec, removed))
                _add(added_total, _hits(rec, added))
        elif kind == KIND_MODAL_EVIDENCE:
            ev = json.loads(text)
            _add(removed_total, _hits(ev, removed))
            _add(added_total, _hits(ev, added))
            bundle = ev.get("petri")
        elif kind == KIND_META:
            cell_dir = path[: -len("/meta.json")]
            rows_path = f"{cell_dir}/responses.jsonl"
            if files.get(rows_path) != KIND_ROWS:
                raise ValueError(f"{cell_dir}: meta.json without responses.jsonl -- the store is incomplete, nothing is proven")
            meta = json.loads(text)
            cell_removed = _hits(meta, removed)
            cell_added = _hits(meta, added)
            for m in (meta.get("served_models") or []):
                served.add(str(m)); observed.add(str(m))
            if meta.get("virtual_model"):
                served.add(str(meta["virtual_model"])); observed.add(str(meta["virtual_model"]))
            rows = 0
            for line in texts[rows_path].splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                rows += 1
                if r.get("model_reported") is not None:
                    reported.add(str(r["model_reported"])); observed.add(str(r["model_reported"]))
                if r.get("virtual_model") is not None:
                    served.add(str(r["virtual_model"])); observed.add(str(r["virtual_model"]))
                _add(cell_removed, _hits(r, removed))
                _add(cell_added, _hits(r, added))
            n_rows += rows
            rel = cell_dir[len(root) + 1:]
            per_cell[rel[len("exp/misalign/"):] if rel.startswith("exp/misalign/") else rel] = {
                "rows": rows, "removed_leg_uses": cell_removed, "added_leg_uses": cell_added, "virtual_model": meta.get("virtual_model"), "served_models": meta.get("served_models")}
            _add(removed_total, cell_removed)
            _add(added_total, cell_added)
        elif kind == KIND_ROWS:
            if files.get(f"{path[: -len('/responses.jsonl')]}/meta.json") != KIND_META:
                raise ValueError(f"{path}: responses.jsonl without meta.json -- the store is incomplete, nothing is proven")
        else:
            raise ValueError(f"{path}: unknown input kind {kind!r}")
    uses = sum(removed_total.values())
    added_uses = sum(added_total.values())
    foreign = sorted(observed - target_set)
    failures: List[str] = []
    if not per_cell:
        failures.append("no cells under exp/misalign -- nothing to prove")
    if uses:
        failures.append(f"the REMOVED leg(s) were reached {uses} time(s): {removed_total}")
    if added_uses:
        failures.append(f"the ADDED leg(s) were reached {added_uses} time(s): {added_total} -- a model this tree's chain introduced served or is named in the store")
    if foreign:
        failures.append(f"{len(foreign)} observed model name(s) are not in the archived row's target set: {foreign[:4]}")
    ok = not failures
    return {"schema": SCHEMA, "ok": ok, "cells": len(per_cell), "n_rows": n_rows, "n_response_evidence": n_evidence,
            "removed_leg": {"uses": uses, "legs": sorted(set(removed)), "per_leg": removed_total},
            "added_legs": {"uses": added_uses, "legs": sorted(set(added)), "per_leg": added_total},
            "per_cell": per_cell, "served_models": sorted(served), "models_reported": sorted(reported),
            "target_models": sorted(target_set), "foreign_models": foreign, "policy": (dict(policy) if policy is not None else None),
            "judge": {"in_attempt": False, "site": JUDGE_SITE},
            "petri_completion_bundle": bundle, "chain": chain_note,
            "basis": ("generation-only store: every JSON string of every cell's meta.json and responses.jsonl, every response_evidence.jsonl record "
                      "(the proxy's vmodel per served response) and modal_evidence.json scanned for each REMOVED and ADDED leg's model id and alias "
                      "(derived from the two trees' manifests); every observed model name held to the archived row's target set; a hit is a use"),
            "inputs": dict(sorted(inputs.items())), "bytes_read": bytes_read, "why": ("; ".join(failures) if failures else None)}


def walk_inputs(root: Path, *, volume_root: str) -> Dict[str, str]:
    """The pulled root's store files as ``{volume path: kind}`` (the CLI's input set, by walking the pulled tree)."""
    root = Path(root)
    mis = root / "exp" / "misalign"
    if not mis.is_dir():
        raise FileNotFoundError(f"{mis}: no misalign store under the pulled root")
    vol = str(volume_root).rstrip("/")
    out: Dict[str, str] = {}
    if (root / "modal_evidence.json").is_file():
        out[f"{vol}/modal_evidence.json"] = KIND_MODAL_EVIDENCE
    for beh_dir in sorted(d for d in mis.iterdir() if d.is_dir()):
        if (beh_dir / "response_evidence.jsonl").is_file():
            out[f"{vol}/{(beh_dir / 'response_evidence.jsonl').relative_to(root).as_posix()}"] = KIND_EVIDENCE
        for cell_dir in sorted(d for d in beh_dir.iterdir() if d.is_dir()):
            meta_p, rows_p = cell_dir / "meta.json", cell_dir / "responses.jsonl"
            if not meta_p.is_file() or not rows_p.is_file():
                raise FileNotFoundError(f"{cell_dir}: a cell without meta.json / responses.jsonl -- the store is incomplete, nothing is proven")
            out[f"{vol}/{meta_p.relative_to(root).as_posix()}"] = KIND_META
            out[f"{vol}/{rows_p.relative_to(root).as_posix()}"] = KIND_ROWS
    return out


def local_reader(root: Path, *, volume_root: str) -> Callable[[str], Optional[str]]:
    """A `read_text` over a pulled root: a volume path under ``volume_root`` -> the local file's text (None when absent)."""
    root = Path(root); vol = str(volume_root).rstrip("/")

    def read(path: str) -> Optional[str]:
        if not str(path).startswith(vol + "/"):
            return None
        p = root / str(path)[len(vol) + 1:]
        return p.read_text(encoding="utf-8") if p.is_file() else None
    return read


def derive(root: Path, *, volume_root: str, removed: Sequence[str], added: Sequence[str], target: Sequence[str], policy: Optional[Mapping[str, Any]] = None,
           chain_note: Optional[str] = None) -> Dict[str, Any]:
    """The proof over a PULLED root (the CLI): the store files by walking the tree, read through :func:`local_reader`, derived by :func:`derive_over`
    under an EXPLICIT policy (removed / added identities and the target set -- derived from the trees by the caller, see `main`)."""
    vol = str(volume_root).rstrip("/")
    files = walk_inputs(root, volume_root=vol)
    if not any(k == KIND_META for k in files.values()):
        files = {k: v for k, v in files.items() if v == KIND_MODAL_EVIDENCE}
    return derive_over(local_reader(root, volume_root=vol), root=vol, files=files, removed=removed, added=added, target=target, policy=policy, chain_note=chain_note)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="the pulled attempt root (holds exp/misalign/...)")
    ap.add_argument("--volume-root", required=True, help="the attempt root's /vol/... spelling (the inputs are recorded under it)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3], help="the repository holding both trees (git objects)")
    ap.add_argument("--prior-dir", required=True, type=Path,
                    help="the prior tree's archived run state (RS/prior/<W7>/, or its plan file) -- the row the consumer evaluates is read from here")
    ap.add_argument("--standup-id", required=True, help="the consuming production row's standup id (the archived plan's row of the same id)")
    ap.add_argument("--prev-w", required=True, help="the prior tree (must be the archived plan's W)")
    ap.add_argument("--this-w", required=True, help="the launch tree")
    ap.add_argument("--chain-note", default=None)
    a = ap.parse_args(argv)
    # fj33: THIS tree's consumer module (scripts/modal/seedrep_carry_over.py) and its imports (src/) -- the policy is the consumer's, one code path
    here = Path(__file__).resolve()
    for extra in (here.parents[3] / "src", here.parents[2] / "modal"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    try:
        import seedrep_carry_over as co  # noqa: PLC0415 -- the consumer: load_prior_source / archived_row / generation_proof_policy
    except ImportError as e:
        sys.stderr.write(f"misalign_generation_proof: this tree's seedrep_carry_over is not importable ({e}) -- the policy cannot be derived; refusing\n")
        return 2
    try:
        prior = co.load_prior_source(a.prior_dir)
        if str(prior.get("W")) != str(a.prev_w):
            raise ValueError(f"--prev-w {str(a.prev_w)[:12]} is not the archived plan's W {str(prior.get('W'))[:12]} ({prior.get('plan_path')})")
        row = co.archived_row(prior, a.standup_id)
        policy = co.generation_proof_policy(row, git=co.default_git(a.repo), prev_W=a.prev_w, this_W=a.this_w)
        rec = derive(a.root, volume_root=a.volume_root.rstrip("/"), removed=policy["removed"], added=policy["added"], target=policy["target"],
                     policy=policy, chain_note=a.chain_note)
        rec = {"kind": co.GENERATION_PROOF_KIND, "ruling": co.GENERATION_PROOF_RULING, **rec}      # the labels the consumer checks, its spelling
    except (OSError, ValueError, KeyError) as e:
        sys.stderr.write(f"misalign_generation_proof: {type(e).__name__}: {e}\n")
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rec, indent=1, sort_keys=True)
    a.out.write_text(text)
    print(f"ok={rec['ok']} cells={rec['cells']} rows={rec['n_rows']} response_evidence={rec['n_response_evidence']} removed_uses={rec['removed_leg']['uses']} "
          f"added_uses={rec['added_legs']['uses']} inputs={len(rec['inputs'])} bytes_read={rec.get('bytes_read')} -> {a.out} sha256={_sha(text.encode('utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
