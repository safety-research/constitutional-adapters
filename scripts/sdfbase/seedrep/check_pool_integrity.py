#!/usr/bin/env python
"""Pool-integrity check for one seed replicate's pool directories.

``build_pools.py verify-nesting`` needs at least two mixed-cell manifests
(the 1k ⊂ 1m ⊂ 10m ⊂ full ladder), which only seed 1 carries; seeds 2 and 3
hold a single ``constv2-mix-full`` cell and would be refused. This check
covers every seed the same way and reports nesting as ``n/a — single mixed
cell`` when the ladder is absent. Everything is re-derived from the pool
FILES, the split artifacts and the corpus finals' own per-record token
counts — never trusted from a manifest alone:

constitution pools (``--pools``, against ``--split``)
- every ``pairs_<cell>.manifest.json``: ``pool_sha256`` == sha256 of the
  jsonl; ``doc_ids_in_order`` == the file's rows; ``n_docs``; every row's
  ``instruction_id`` is the cell; at most one truncated row and it is the
  manifest's ``truncation.doc_id``; ``heldout_split_sha`` == sha256 of the
  split file;
- ``constv2-mix-full``: doc set == corpus − seed held-out EXACTLY, whole
  docs (no truncation); the other mixed cells ⊆ corpus − held-out with
  ``official_tokens == budget``;
- ``constv2-heldout``: ids == the split's ``eval_subset.ids`` in order;
- token totals: ``per_tokenizer_cumulative_totals`` recomputed from the
  corpus records' ``token_counts`` (the truncated doc contributing its
  recorded ``tokens_taken_per_tokenizer``), ``min == official_tokens``;
- nesting: ``verify_mix_nesting`` when >= 2 mixed manifests, else n/a.

control pools (``--ctl2``, against ``--ctl2-split``)
- the same file/manifest agreement (ids wear the ``ctl2-`` prefix);
- ``constv2ctl2-full`` ⊂ control corpus − control held-out;
  ``official_tokens == budget == budget_source.official_tokens ==`` the
  constitution ``constv2-mix-full`` manifest's ``official_tokens`` (the
  budget-matched contract), and ``budget_source.manifest_sha256`` is the
  sha of that manifest as it stands; prefix ∪ remainder == the control
  train side, prefix ∩ remainder == ∅, remainder ∩ held-out == ∅;
- ``constv2ctl2-heldout``: ids == ``ctl2-`` + the control split's eval ids;
- ``build_ctl2_pools.verify_pools`` (the campaign's own byte/row/id check).

Exit 0 with every check named, or 2 with the failures listed. Read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SDFBASE = _HERE.parent
if str(_SDFBASE) not in sys.path:
    sys.path.insert(0, str(_SDFBASE))

import build_ctl2_pools as b2  # noqa: E402
import build_heldout_split as hsplit  # noqa: E402
import build_pools as bp  # noqa: E402

NESTING_NA = "n/a — single mixed cell"


# --------------------------------------------------------------------------
# Nesting: strict prefix containment, or skip-aware across a boundary skip
# --------------------------------------------------------------------------

def nesting_skip_aware(manifests: list[dict]) -> list[dict]:
    """Doc-id containment across consecutive mixed cells that TOLERATES the
    boundary skip-forward rule, exactly: a cell's whole docs W are a
    prefix of the next cell's order, followed IN ORDER by the docs the
    smaller cell skipped at its boundary (``boundary_skips``), followed by
    its truncated boundary doc (``truncation.doc_id``), which the bigger
    cell carries whole. ``build_pools.verify_mix_nesting`` refuses any dir
    with a skip by design ("inspect the manifests directly"); this is that
    inspection, as code. ``manifests`` in nesting order (ascending budget).
    Returns one verdict dict per consecutive pair; raises SamplerError on
    the first violation."""
    out = []
    for small, big in zip(manifests, manifests[1:]):
        s_ids = list(small["doc_ids_in_order"])
        b_ids = list(big["doc_ids_in_order"])
        trunc = (small.get("truncation") or {}).get("doc_id")
        skipped = [s["doc_id"] for s in (small.get("boundary_skips") or [])]
        if trunc is not None:
            if not s_ids or s_ids[-1] != trunc:
                raise bp.SamplerError(
                    f"nesting: {small['cell']} truncation doc {trunc!r} is not "
                    "its last doc")
            whole = s_ids[:-1]
        else:
            if skipped:
                raise bp.SamplerError(
                    f"nesting: {small['cell']} records boundary skips "
                    f"{skipped} but no truncated boundary doc")
            whole = s_ids
        expected = whole + skipped + ([trunc] if trunc is not None else [])
        if len(b_ids) < len(expected):
            raise bp.SamplerError(
                f"nesting: {big['cell']} lists {len(b_ids)} docs but "
                f"{small['cell']} needs {len(expected)} (whole + skipped + "
                "boundary) — the larger-budget cell is shorter")
        if b_ids[:len(expected)] != expected:
            k = next(i for i, (a, b) in enumerate(zip(expected, b_ids)) if a != b)
            raise bp.SamplerError(
                f"nesting: {small['cell']} is NOT contained in {big['cell']} "
                f"(skip-aware; first divergence at position {k}: expected "
                f"{expected[k]!r}, {big['cell']} has {b_ids[k]!r})")
        out.append({"small": small["cell"], "big": big["cell"],
                    "whole_docs": len(whole), "skipped": skipped,
                    "boundary_doc": trunc})
    return out


def verify_nesting(pools_dir: Path) -> tuple[str, dict | None]:
    """The nesting verdict for a pools dir: ``(label, details)``.

    Fewer than two mixed-cell manifests → ``(NESTING_NA, None)``. Otherwise
    the campaign's strict check (``verify_mix_nesting``) first; when it
    refuses ONLY because a cell carries boundary skips, the skip-aware
    containment (:func:`nesting_skip_aware`) decides and the label says so.
    Any other refusal propagates as SamplerError."""
    pools_dir = Path(pools_dir)
    names = [c for c, _ in bp.MIX_CELLS
             if (pools_dir / f"pairs_{c}.manifest.json").is_file()]
    if len(names) < 2:
        return NESTING_NA, None
    try:
        strict = bp.verify_mix_nesting(pools_dir)
        return " ⊂ ".join(strict), {"mode": "strict", "order": strict}
    except bp.SamplerError as e:
        if "boundary_skips" not in str(e):
            raise
    manifests = [load_manifest(pools_dir, c) for c in names]
    pairs = nesting_skip_aware(manifests)
    skips = {p["small"]: p["skipped"] for p in pairs if p["skipped"]}
    label = " ⊂ ".join(names) + " (skip-aware: " + "; ".join(
        f"{cell} skipped {ids} at its boundary (D9)" for cell, ids in skips.items()
    ) + ")"
    return label, {"mode": "skip-aware", "order": names, "pairs": pairs}
CTL2_CONTROL_ADVICE = (hsplit.GENERATOR_ROOT
                       / "data/control/final-control-scale-20260903-r2-advice.jsonl")
CTL2_CONTROL_STORIES = (hsplit.GENERATOR_ROOT
                        / "data/control/final-control-scale-20260903-r2-stories.jsonl")


class IntegrityError(RuntimeError):
    """A malformed input (a check that cannot even be evaluated)."""


def _tilde(p) -> str:
    s = str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home + "/") else s


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail=None) -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def corpus_index(advice_final, stories_final) -> dict[str, dict]:
    """doc_id -> {"family", "counts"} from the corpus finals' own
    per-record ``token_counts`` (the authoritative per-record accounting
    the builders summed); text is dropped immediately."""
    idx: dict[str, dict] = {}
    for path, fam in ((advice_final, "advice"), (stories_final, "stories")):
        for rec in hsplit.load_jsonl(Path(path)):
            did = hsplit.doc_id(rec)
            if did in idx:
                raise IntegrityError(f"duplicate doc id {did} in the corpus")
            idx[did] = {"family": fam,
                        "counts": hsplit.record_token_counts(rec)}
    return idx


def read_rows(pool_path) -> list[dict]:
    """The light per-row view (doc_id, instruction_id, truncated, family)."""
    rows: list[dict] = []
    with open(pool_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({"line": lineno, "doc_id": r.get("doc_id"),
                         "instruction_id": r.get("instruction_id"),
                         "truncated": bool(r.get("truncated")),
                         "family": r.get("family")})
    return rows


def load_manifest(pools_dir: Path, cell: str) -> dict:
    mp = pools_dir / f"pairs_{cell}.manifest.json"
    if not mp.is_file():
        raise IntegrityError(f"{mp} missing")
    return json.loads(mp.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_file_agreement(rep: Report, pools_dir: Path, cell: str,
                         manifest: dict, rows: list[dict], *,
                         id_prefix: str = "") -> None:
    pool = pools_dir / f"pairs_{cell}.jsonl"
    tag = f"{pools_dir.name}/{cell}"
    rep.add(f"{tag}: pool_sha256 == sha256(jsonl)",
            hsplit.sha256_file(pool) == manifest.get("pool_sha256"),
            {"manifest": manifest.get("pool_sha256")})
    ids = [r["doc_id"] for r in rows]
    rep.add(f"{tag}: doc_ids_in_order == file rows",
            ids == manifest.get("doc_ids_in_order"),
            {"rows": len(ids), "manifest": len(manifest.get("doc_ids_in_order") or [])})
    rep.add(f"{tag}: n_docs == rows", manifest.get("n_docs") == len(ids),
            {"n_docs": manifest.get("n_docs"), "rows": len(ids)})
    bad_cell = [r["line"] for r in rows if r["instruction_id"] != cell]
    rep.add(f"{tag}: every row instruction_id == cell", not bad_cell,
            {"bad_lines": bad_cell[:5]})
    if id_prefix:
        bad_prefix = [r["doc_id"] for r in rows
                      if not str(r["doc_id"]).startswith(id_prefix)]
        rep.add(f"{tag}: every doc_id wears the {id_prefix!r} prefix",
                not bad_prefix, {"first": bad_prefix[:5]})
    truncated = [r["doc_id"] for r in rows if r["truncated"]]
    trunc = manifest.get("truncation") or {}
    rep.add(f"{tag}: at most one truncated row, == manifest truncation",
            len(truncated) <= 1
            and (truncated[:1] == ([trunc["doc_id"]] if trunc else [])),
            {"rows_truncated": truncated[:3], "manifest": trunc.get("doc_id")})
    if manifest.get("budget") is None:
        rep.add(f"{tag}: whole-order cell has no truncation",
                not truncated and not trunc, None)


def recompute_totals(rep: Report, tag: str, manifest: dict, rows: list[dict],
                     index: dict[str, dict], *, id_prefix: str = "") -> None:
    """``per_tokenizer_cumulative_totals`` from the corpus records' counts
    (whole docs) + the truncated doc's recorded ``tokens_taken``."""
    totals = {k: 0 for k in hsplit.TOK_NAMES}
    trunc = manifest.get("truncation") or {}
    missing = []
    for r in rows:
        raw = str(r["doc_id"])[len(id_prefix):] if id_prefix else str(r["doc_id"])
        rec = index.get(raw)
        if rec is None:
            missing.append(r["doc_id"])
            continue
        if r["truncated"] and trunc.get("doc_id") == r["doc_id"]:
            counts = trunc.get("tokens_taken_per_tokenizer") or {}
        else:
            counts = rec["counts"]
        for k in hsplit.TOK_NAMES:
            totals[k] += int(counts.get(k, 0))
    rep.add(f"{tag}: every row's doc is a corpus doc", not missing,
            {"missing": missing[:5]})
    rep.add(f"{tag}: per_tokenizer totals recomputed from corpus counts",
            totals == manifest.get("per_tokenizer_cumulative_totals"),
            {"recomputed": totals,
             "manifest": manifest.get("per_tokenizer_cumulative_totals")})
    official = min(totals.values())
    rep.add(f"{tag}: min(totals) == official_tokens",
            official == manifest.get("official_tokens"),
            {"recomputed": official, "manifest": manifest.get("official_tokens")})
    if manifest.get("budget") is not None:
        rep.add(f"{tag}: official_tokens == budget",
                manifest.get("official_tokens") == manifest.get("budget"),
                {"official": manifest.get("official_tokens"),
                 "budget": manifest.get("budget")})


def check_constitution(rep: Report, pools_dir: Path, split_path: Path,
                       index: dict[str, dict]) -> dict:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split_sha = hsplit.sha256_file(split_path)
    held = {i for fam in ("advice", "stories") for i in split[fam]["ids"]}
    corpus_ids = set(index)
    rep.add(f"{pools_dir.name}: split held-out ids ⊆ corpus", held <= corpus_ids,
            {"outside": sorted(held - corpus_ids)[:5]})
    train_ids = corpus_ids - held

    mixed_present = [c for c, _ in bp.MIX_CELLS
                     if (pools_dir / f"pairs_{c}.manifest.json").is_file()]
    heldout_present = (pools_dir / f"pairs_{bp.HELDOUT_CELL}.manifest.json").is_file()
    rep.add(f"{pools_dir.name}: constv2-mix-full and constv2-heldout present",
            "constv2-mix-full" in mixed_present and heldout_present,
            {"mixed": mixed_present, "heldout": heldout_present})

    out: dict = {"cells": {}}
    for cell in mixed_present:
        m = load_manifest(pools_dir, cell)
        rows = read_rows(pools_dir / f"pairs_{cell}.jsonl")
        tag = f"{pools_dir.name}/{cell}"
        check_file_agreement(rep, pools_dir, cell, m, rows)
        rep.add(f"{tag}: heldout_split_sha == sha256(split)",
                m.get("heldout_split_sha") == split_sha, None)
        ids = {r["doc_id"] for r in rows}
        rep.add(f"{tag}: doc set ⊆ corpus − held-out", ids <= train_ids,
                {"held_out_inside": sorted(ids & held)[:5],
                 "unknown": sorted(ids - corpus_ids)[:5]})
        if cell == "constv2-mix-full":
            rep.add(f"{tag}: doc set == corpus − held-out EXACTLY",
                    ids == train_ids,
                    {"missing_from_pool": len(train_ids - ids),
                     "extra_in_pool": len(ids - train_ids)})
        recompute_totals(rep, tag, m, rows, index)
        out["cells"][cell] = {"n_docs": len(rows),
                              "official_tokens": m.get("official_tokens"),
                              "pool_sha256": m.get("pool_sha256")}

    if heldout_present:
        cell = bp.HELDOUT_CELL
        m = load_manifest(pools_dir, cell)
        rows = read_rows(pools_dir / f"pairs_{cell}.jsonl")
        tag = f"{pools_dir.name}/{cell}"
        check_file_agreement(rep, pools_dir, cell, m, rows)
        rep.add(f"{tag}: heldout_split_sha == sha256(split)",
                m.get("heldout_split_sha") == split_sha, None)
        rep.add(f"{tag}: ids == split eval_subset.ids (ordered)",
                [r["doc_id"] for r in rows] == list(split["eval_subset"]["ids"]),
                {"rows": len(rows), "eval_n": split["eval_subset"]["n"]})
        recompute_totals(rep, tag, m, rows, index)
        out["cells"][cell] = {"n_docs": len(rows),
                              "official_tokens": m.get("official_tokens"),
                              "pool_sha256": m.get("pool_sha256")}

    try:
        label, details = verify_nesting(pools_dir)
        rep.add(f"{pools_dir.name}: nesting (doc-id containment)", True,
                {"label": label, "details": details,
                 "mixed_cells": mixed_present})
        out["nesting"] = label
    except bp.SamplerError as e:
        rep.add(f"{pools_dir.name}: nesting (doc-id containment)", False,
                {"error": str(e), "mixed_cells": mixed_present})
        out["nesting"] = f"FAILED: {e}"
    out["heldout_docs"] = len(held)
    out["train_docs"] = len(train_ids)
    return out


def check_control(rep: Report, ctl2_dir: Path, ctl2_split_path: Path,
                  index: dict[str, dict], const_pools_dir: Path) -> dict:
    split = json.loads(ctl2_split_path.read_text(encoding="utf-8"))
    split_sha = hsplit.sha256_file(ctl2_split_path)
    prefix = b2.CTL2_ID_PREFIX
    held = {i for fam in ("advice", "stories") for i in split[fam]["ids"]}
    corpus_ids = set(index)
    rep.add(f"{ctl2_dir.name}: control split held-out ids ⊆ control corpus",
            held <= corpus_ids, {"outside": sorted(held - corpus_ids)[:5]})
    train_ids = corpus_ids - held
    train_pref = {prefix + i for i in train_ids}
    held_pref = {prefix + i for i in held}

    const_full = load_manifest(const_pools_dir, b2.BUDGET_CELL)
    const_full_path = const_pools_dir / f"pairs_{b2.BUDGET_CELL}.manifest.json"
    out: dict = {"cells": {}}

    # the campaign's own byte/row/id verify
    try:
        v = b2.verify_pools(ctl2_dir)
        rep.add(f"{ctl2_dir.name}: build_ctl2_pools.verify_pools", True,
                {c: r["n_docs"] for c, r in v.items()})
    except bp.SamplerError as e:
        rep.add(f"{ctl2_dir.name}: build_ctl2_pools.verify_pools", False,
                {"error": str(e)})

    cell = b2.FULL_CELL
    m = load_manifest(ctl2_dir, cell)
    rows = read_rows(ctl2_dir / f"pairs_{cell}.jsonl")
    tag = f"{ctl2_dir.name}/{cell}"
    check_file_agreement(rep, ctl2_dir, cell, m, rows, id_prefix=prefix)
    rep.add(f"{tag}: heldout_split_sha == sha256(ctl2 split)",
            m.get("heldout_split_sha") == split_sha, None)
    ids = {r["doc_id"] for r in rows}
    rep.add(f"{tag}: doc set ⊆ control corpus − control held-out",
            ids <= train_pref,
            {"held_out_inside": sorted(ids & held_pref)[:5],
             "unknown": sorted(ids - train_pref - held_pref)[:5]})
    recompute_totals(rep, tag, m, rows, index, id_prefix=prefix)
    budget = m.get("budget")
    bsrc = m.get("budget_source") or {}
    rep.add(f"{tag}: official_tokens == budget == budget_source == "
            f"constitution {b2.BUDGET_CELL} official_tokens",
            (m.get("official_tokens") == budget
             == bsrc.get("official_tokens") == const_full.get("official_tokens")),
            {"official": m.get("official_tokens"), "budget": budget,
             "budget_source": bsrc.get("official_tokens"),
             "constitution_manifest": const_full.get("official_tokens")})
    rep.add(f"{tag}: budget_source.manifest_sha256 == sha256(constitution "
            "mix-full manifest as it stands)",
            bsrc.get("manifest_sha256") == hsplit.sha256_file(const_full_path),
            {"budget_source": bsrc.get("manifest_sha256"),
             "now": hsplit.sha256_file(const_full_path)})
    rep.add(f"{tag}: budget_source.pool_sha256 == constitution mix-full "
            "pool_sha256",
            bsrc.get("pool_sha256") == const_full.get("pool_sha256"), None)
    rem = set((m.get("remainder") or {}).get("doc_ids_in_order") or [])
    rep.add(f"{tag}: prefix ∩ remainder == ∅", not (ids & rem),
            {"overlap": len(ids & rem)})
    rep.add(f"{tag}: prefix ∪ remainder == control train side",
            (ids | rem) == train_pref,
            {"missing": len(train_pref - ids - rem),
             "extra": len((ids | rem) - train_pref)})
    rep.add(f"{tag}: remainder ∩ held-out == ∅", not (rem & held_pref),
            {"overlap": len(rem & held_pref)})
    out["cells"][cell] = {"n_docs": len(rows),
                          "official_tokens": m.get("official_tokens"),
                          "budget": budget,
                          "remainder_docs": len(rem),
                          "remainder_official_tokens":
                              (m.get("remainder") or {}).get("official_tokens"),
                          "pool_sha256": m.get("pool_sha256")}

    cell = b2.HELDOUT_CELL
    m = load_manifest(ctl2_dir, cell)
    rows = read_rows(ctl2_dir / f"pairs_{cell}.jsonl")
    tag = f"{ctl2_dir.name}/{cell}"
    check_file_agreement(rep, ctl2_dir, cell, m, rows, id_prefix=prefix)
    rep.add(f"{tag}: heldout_split_sha == sha256(ctl2 split)",
            m.get("heldout_split_sha") == split_sha, None)
    rep.add(f"{tag}: ids == ctl2- + control split eval_subset.ids (ordered)",
            [r["doc_id"] for r in rows]
            == [prefix + i for i in split["eval_subset"]["ids"]], None)
    recompute_totals(rep, tag, m, rows, index, id_prefix=prefix)
    out["cells"][cell] = {"n_docs": len(rows),
                          "official_tokens": m.get("official_tokens"),
                          "pool_sha256": m.get("pool_sha256")}
    out["heldout_docs"] = len(held)
    out["train_docs"] = len(train_ids)
    return out


def run(*, pools, ctl2, split, ctl2_split, advice_final=None,
        stories_final=None, control_advice=None, control_stories=None) -> dict:
    """Every check over one seed's two pool dirs; returns the report dict
    (``ok``, ``checks``, per-dir summaries)."""
    pools, ctl2 = Path(pools), Path(ctl2)
    split, ctl2_split = Path(split), Path(ctl2_split)
    rep = Report()
    const_index = corpus_index(advice_final or hsplit.ADVICE_FINAL,
                               stories_final or hsplit.STORIES_FINAL)
    const = check_constitution(rep, pools, split, const_index)
    del const_index
    ctl_index = corpus_index(control_advice or CTL2_CONTROL_ADVICE,
                             control_stories or CTL2_CONTROL_STORIES)
    ctl = check_control(rep, ctl2, ctl2_split, ctl_index, pools)
    return {
        "pools": _tilde(pools), "ctl2": _tilde(ctl2),
        "split": _tilde(split), "split_sha256": hsplit.sha256_file(split),
        "ctl2_split": _tilde(ctl2_split),
        "ctl2_split_sha256": hsplit.sha256_file(ctl2_split),
        "constitution": const, "control": ctl,
        "nesting": const["nesting"],
        "n_checks": len(rep.checks),
        "failures": rep.failures(),
        "checks": rep.checks,
        "ok": rep.ok,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--ctl2", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--ctl2-split", type=Path, required=True)
    ap.add_argument("--advice-final", type=Path, default=hsplit.ADVICE_FINAL)
    ap.add_argument("--stories-final", type=Path, default=hsplit.STORIES_FINAL)
    ap.add_argument("--control-advice", type=Path, default=CTL2_CONTROL_ADVICE)
    ap.add_argument("--control-stories", type=Path,
                    default=CTL2_CONTROL_STORIES)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        rep = run(pools=args.pools, ctl2=args.ctl2, split=args.split,
                  ctl2_split=args.ctl2_split, advice_final=args.advice_final,
                  stories_final=args.stories_final,
                  control_advice=args.control_advice,
                  control_stories=args.control_stories)
    except (IntegrityError, bp.SamplerError, hsplit.SplitError, OSError,
            KeyError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, indent=1, ensure_ascii=False)
                               + "\n", encoding="utf-8")
    for c in rep["checks"]:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['name']}")
    print(f"nesting: {rep['nesting']}")
    if not rep["ok"]:
        for c in rep["failures"]:
            print(f"REFUSED: {c['name']}: {c['detail']}", file=sys.stderr)
        return 2
    print(f"pool integrity OK ({rep['n_checks']} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
