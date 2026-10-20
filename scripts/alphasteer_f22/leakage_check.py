#!/usr/bin/env python
"""alphasteer-full22 Phase 0 re-expressed in-repo: the ONLY producer of `leakage_report.json`
(plan §5.4, §8; decisions D19, D29).

Contract
--------
Inputs (pinned, read never retyped):
  * the 259-goal confirmatory universe — `jailbreak_run._eval_goals(cfg, confirmatory=True)` over the
    campaign config (`configs/alphasteer_f22_serve_<target>.yaml`; every full22 config carries the same
    `datasets` block, so the universe is target-independent — asserted, not assumed, when several
    configs are given);
  * the authors' harmful fit list `harmful_alphasteer.json` (2,000 prompts; sha256 `7c6983fc…`, asserted).
Method (the method Phase 0 ACTUALLY ran — NOT MinHash, §5.4):
  1. `normalise_prompt` — NFKC -> lower -> `[^\\w\\s]` -> space -> `\\s+` collapse -> strip. Vendored
     VERBATIM from `src/sisteer/extraction/alphasteer_screen.py` on branch `alphasteer-multilayer`
     (that module is not on this tree yet); its `inspect.getsource` sha256 is pinned
     (`NORMALISE_PROMPT_SOURCE_SHA256`) and RECORDED in the output, so an edit to the function
     invalidates the artifact loudly (D29). When `sisteer.extraction.alphasteer_screen` IS importable,
     its function is used instead and MUST hash to the same pin, else refuse.
  2. `exact`: goals whose normalised text equals some normalised fit prompt (sorted goal_ids);
  3. `near`: goals NOT in `exact` with max word-3-gram Jaccard >= 0.8 against ANY fit prompt, exhaustive
     over all 259 x 2,000 pairs (`{goal_id: max_jaccard}`, rounded to 3 dp as the frozen artifact is);
  4. `total` = sorted(exact ∪ near); `n_goals` = 259; `frac` = len(total)/n_goals.
Output: the frozen file's FIVE keys with their names and types unchanged (every existing reader resolves
`total`), plus ADDED provenance beside them: `normalise_prompt_source_sha256`,
`normalise_prompt_vendored_from`, `harmful_fit_list` {path, sha256, n}, `goal_universe` {n, sha256 of
the sorted ids, config paths}, `method`, `per_goal` evidence (per contaminated goal: kind, max_jaccard,
fit row index), `written_at`, and a payload `sha256`.
Acceptance (plan §5.4 item 4): `--frozen <path>` compares `exact` / `near` / `total` against the frozen
artifact id-for-id BEFORE any write; a diff is a STOP (exit 2, the diff printed), never a restatement.
The clean subset is DERIVED by consumers as universe minus `total` (`clean_goal_ids(universe, report)`
here) — the file carries no clean list and no literal 128 anywhere.

usage: leakage_check.py --config configs/alphasteer_f22_serve_gemma4e4b.yaml [--config ...]
                        --harmful <harmful_alphasteer.json>
                        --out <leakage_report.json> --frozen <the frozen leakage_report.json, or '' for a NEW freeze>
                        [--check]   # compare only, write nothing
The harmful fit list (the AlphaSteer authors' harmful_alphasteer.json, 2,000 prompts, sha256 `7c6983fc…`) is NOT
distributed with this repository: --harmful names your own copy, and its sha256 is asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO / "src"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "alphasteer_f22_leakage/1"
JACCARD_THRESHOLD = 0.8
NGRAM_N = 3
NEAR_ROUND_DP = 3
#: sha256 of the authors' harmful fit list (plan §5.4 item 2; 2,000 rows)
HARMFUL_FIT_LIST_SHA256 = "7c6983fca61e13181613c14b18753ecae10cfd2b156f0a4b0104136fb641c5cf"
HARMFUL_FIT_LIST_N = 2000
#: the frozen artifact's own identity (plan §5.4): the acceptance test reproduces its ids exactly
FROZEN_REPORT_SHA256 = "9abd0d24d3baea06764df483fecd737deda715a774ffb81abe0010047bda549e"
FROZEN_KEYS = ("exact", "near", "total", "n_goals", "frac")
#: where `normalise_prompt` was vendored from, verbatim (branch alphasteer-multilayer)
NORMALISE_PROMPT_VENDORED_FROM = {
    "branch": "alphasteer-multilayer",
    "commit": "b089309dfbadf6d90077fc904b7404650a49cfcf",
    "path": "src/sisteer/extraction/alphasteer_screen.py",
    "symbol": "normalise_prompt",
}
#: sha256 of `inspect.getsource(normalise_prompt)` — the D29 identity. Pinned against the branch's own
#: def block (byte-equal, verified 2026-09-09); the output records the sha of the function ACTUALLY
#: used and the script refuses when it differs.
NORMALISE_PROMPT_SOURCE_SHA256 = "26d32e573d8548fae7b693addcc9230c6b28798404627aea59de957a5b7bfb94"
REPO_MODULE = "sisteer.extraction.alphasteer_screen"


class Refusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSING: {msg}")


# --------------------------------------------------------------------------- the ONE normalisation
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalise_prompt(text: str) -> str:
    """The ONE prompt identity: NFKC -> lowercase -> punctuation to space ->
    whitespace collapse. Shared by the leakage check (plan Phase 0) and the
    split below, so the two can never disagree about what "the same
    prompt" means."""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def normaliser_source_sha256(fn=normalise_prompt) -> str:
    """sha256 of the function's source text (`inspect.getsource`) — the D29 identity."""
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def resolve_normaliser(*, pin: str = NORMALISE_PROMPT_SOURCE_SHA256) -> Tuple[Any, Dict[str, Any]]:
    """The normaliser in use + its provenance block: the repo module's when importable, else the
    vendored copy; either way its source sha must equal the pin, else refuse (D29)."""
    fn, origin = normalise_prompt, "vendored"
    try:
        import importlib
        mod = importlib.import_module(REPO_MODULE)
        fn, origin = getattr(mod, "normalise_prompt"), f"imported:{REPO_MODULE}"
    except Exception:  # noqa: BLE001 — the module is not on this tree yet; the vendored copy stands
        pass
    sha = normaliser_source_sha256(fn)
    if sha != pin:
        raise Refusal(f"normalise_prompt source sha256 {sha[:16]} ({origin}) != pinned {pin[:16]} — the "
                      "normalisation changed; the frozen leakage_report.json would be re-partitioned "
                      "silently. Re-pin deliberately (and re-freeze the artifact) or restore the function.")
    return fn, {"origin": origin, "source_sha256": sha, "pinned_source_sha256": pin,
                "vendored_from": dict(NORMALISE_PROMPT_VENDORED_FROM)}


# --------------------------------------------------------------------------- similarity
def word_ngrams(norm: str, n: int = NGRAM_N) -> Set[Tuple[str, ...]]:
    w = norm.split()
    if len(w) < n:
        return set()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --------------------------------------------------------------------------- inputs
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_harmful_fit_list(path: Path, *, expect_sha: Optional[str] = HARMFUL_FIT_LIST_SHA256,
                          expect_n: Optional[int] = HARMFUL_FIT_LIST_N) -> List[str]:
    """The 2,000 prompts, sha-asserted (a different file is a different experiment)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise Refusal(f"harmful fit list not found at {p}: this repository does not distribute it (the AlphaSteer "
                      "authors' harmful_alphasteer.json, 2,000 prompts, sha256 7c6983fc…); pass --harmful <your copy>")
    sha = sha256_file(p)
    if expect_sha is not None and sha != expect_sha:
        raise Refusal(f"{p}: sha256 {sha[:16]} != pinned {expect_sha[:16]} (plan §5.4 item 2) — not the "
                      "authors' harmful_alphasteer.json this artifact is defined over")
    rows = json.loads(p.read_text())
    if not isinstance(rows, list) or not all(isinstance(r, str) for r in rows):
        raise Refusal(f"{p}: expected a JSON list of prompt strings")
    if expect_n is not None and len(rows) != expect_n:
        raise Refusal(f"{p}: {len(rows)} rows != {expect_n}")
    return rows


def goal_universe(config_paths: Sequence[Path]) -> Dict[str, str]:
    """`{goal_id: goal_text}` for the 259-goal confirmatory set via `jailbreak_run._eval_goals`;
    several configs must agree exactly (same ids, same texts)."""
    import yaml
    import jailbreak_run as jr

    out: Optional[Dict[str, str]] = None
    for cp in config_paths:
        cfg = yaml.safe_load(Path(cp).expanduser().read_text())
        goals = jr._eval_goals(cfg, confirmatory=True)
        this = {g.goal_id: g.text for g in goals}
        if len(this) != len(goals):
            raise Refusal(f"{cp}: duplicate goal ids in the eval set")
        if out is None:
            out = this
        elif this != out:
            raise Refusal(f"{cp}: goal universe differs from {config_paths[0]} — the configs do not share "
                          "one datasets block; the leakage report is defined over ONE universe")
    if not out:
        raise Refusal("no config given")
    return out


def universe_sha256(goal_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(goal_ids)).encode()).hexdigest()


# --------------------------------------------------------------------------- the check
def leakage(goal_texts: Mapping[str, str], fit_prompts: Sequence[str], *, normaliser=normalise_prompt,
            threshold: float = JACCARD_THRESHOLD, n: int = NGRAM_N) -> Dict[str, Any]:
    """The five frozen keys + `per_goal` evidence. `near` values are rounded to 3 dp (the frozen file's
    precision). Exhaustive over every (goal, fit prompt) pair — no sketching."""
    fit_norm = [normaliser(p) for p in fit_prompts]
    fit_index: Dict[str, int] = {}
    for i, t in enumerate(fit_norm):
        fit_index.setdefault(t, i)           # first row index carrying this normalised text
    fit_grams = [word_ngrams(t, n) for t in fit_norm]
    exact: List[str] = []
    near: Dict[str, float] = {}
    per_goal: Dict[str, Dict[str, Any]] = {}
    for gid in sorted(goal_texts):
        gn = normaliser(goal_texts[gid])
        if gn in fit_index:
            exact.append(gid)
            per_goal[gid] = {"kind": "exact", "fit_row": fit_index[gn], "max_jaccard": 1.0}
            continue
        gg = word_ngrams(gn, n)
        best, best_i = 0.0, -1
        for i, fg in enumerate(fit_grams):
            j = jaccard(gg, fg)
            if j > best:
                best, best_i = j, i
        if best >= threshold:
            near[gid] = round(best, NEAR_ROUND_DP)
            per_goal[gid] = {"kind": "near", "fit_row": best_i, "max_jaccard": round(best, 6)}
    total = sorted(set(exact) | set(near))
    n_goals = len(goal_texts)
    return {"exact": sorted(exact), "near": dict(sorted(near.items())), "total": total,
            "n_goals": n_goals, "frac": (len(total) / n_goals) if n_goals else 0.0,
            "per_goal": per_goal}


def clean_goal_ids(universe: Iterable[str], report: Mapping[str, Any]) -> List[str]:
    """The clean subset every consumer derives: universe minus `report['total']` (never a literal)."""
    total = set(report["total"])
    uni = set(universe)
    stray = total - uni
    if stray:
        raise Refusal(f"{len(stray)} contaminated ids are not in the goal universe (e.g. {sorted(stray)[:3]}) — "
                      "the report and the universe disagree; not a clean subset")
    return sorted(uni - total)


def diff_against_frozen(ours: Mapping[str, Any], frozen: Mapping[str, Any]) -> List[str]:
    """Human-readable differences over `exact` / `near` / `total` / `n_goals`; empty == reproduces."""
    problems: List[str] = []
    for key in ("exact", "total"):
        a, b = set(ours.get(key, [])), set(frozen.get(key, []))
        if a != b:
            problems.append(f"{key}: ours-only {sorted(a - b)} frozen-only {sorted(b - a)} "
                            f"(ours {len(a)}, frozen {len(b)})")
        elif list(ours.get(key, [])) != sorted(a):
            problems.append(f"{key}: not sorted")
    on, fn = dict(ours.get("near", {})), dict(frozen.get("near", {}))
    if set(on) != set(fn):
        problems.append(f"near ids: ours-only {sorted(set(on) - set(fn))} frozen-only {sorted(set(fn) - set(on))}")
    for gid in sorted(set(on) & set(fn)):
        if round(float(on[gid]), NEAR_ROUND_DP) != round(float(fn[gid]), NEAR_ROUND_DP):
            problems.append(f"near[{gid}]: ours {on[gid]} frozen {fn[gid]}")
    if int(ours.get("n_goals", -1)) != int(frozen.get("n_goals", -2)):
        problems.append(f"n_goals: ours {ours.get('n_goals')} frozen {frozen.get('n_goals')}")
    return problems


def _payload_sha(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, indent=1, sort_keys=True).encode()).hexdigest()


def build_report(goal_texts: Mapping[str, str], fit_prompts: Sequence[str], *, harmful_path: Path,
                 config_paths: Sequence[Path], normaliser_block: Optional[Dict[str, Any]] = None,
                 normaliser=None) -> Dict[str, Any]:
    if normaliser is None:
        normaliser, normaliser_block = resolve_normaliser()
    core = leakage(goal_texts, fit_prompts, normaliser=normaliser)
    rep: Dict[str, Any] = {k: core[k] for k in FROZEN_KEYS}
    hp = Path(harmful_path).expanduser()
    rep.update({
        "schema": SCHEMA,
        "method": {"normalisation": "normalise_prompt (NFKC -> lower -> [^\\w\\s] -> space -> \\s+ collapse -> strip)",
                   "exact": "normalised goal text == some normalised fit prompt",
                   "near": f"not exact AND max over fit prompts of word-{NGRAM_N}-gram Jaccard >= {JACCARD_THRESHOLD}, "
                           "exhaustive over every pair (NOT MinHash)",
                   "near_value": f"the max Jaccard, rounded to {NEAR_ROUND_DP} dp",
                   "total": "sorted(exact ∪ near)", "clean_subset": "DERIVED by consumers: universe minus total"},
        "normalise_prompt_source_sha256": (normaliser_block or {}).get("source_sha256", normaliser_source_sha256(normaliser)),
        "normalise_prompt_origin": (normaliser_block or {}).get("origin", "vendored"),
        "normalise_prompt_vendored_from": dict(NORMALISE_PROMPT_VENDORED_FROM),
        "harmful_fit_list": {"path": str(hp), "sha256": sha256_file(hp) if hp.is_file() else None, "n": len(fit_prompts)},
        "goal_universe": {"n": len(goal_texts), "sha256": universe_sha256(goal_texts.keys()),
                          "configs": [str(Path(c).expanduser()) for c in config_paths],
                          "source": "jailbreak_run._eval_goals(cfg, confirmatory=True)"},
        "per_goal": core["per_goal"],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    rep["sha256"] = _payload_sha(rep)
    return rep


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", action="append", required=True, help="campaign config(s); several must agree")
    ap.add_argument("--harmful", required=True, help="the authors' harmful_alphasteer.json (sha-asserted)")
    ap.add_argument("--out", default=None, help="where to write leakage_report.json (omit with --check)")
    ap.add_argument("--frozen", required=True,
                    help="the frozen artifact to reproduce id-for-id before writing; '' to skip (a NEW freeze)")
    ap.add_argument("--check", action="store_true", help="compare against --frozen only; write nothing")
    args = ap.parse_args(argv)
    if not args.check and not args.out:
        raise Refusal("--out is required unless --check")

    normaliser, nblock = resolve_normaliser()
    fit = load_harmful_fit_list(Path(args.harmful))
    goals = goal_universe([Path(c) for c in args.config])
    rep = build_report(goals, fit, harmful_path=Path(args.harmful), config_paths=[Path(c) for c in args.config],
                       normaliser_block=nblock, normaliser=normaliser)
    print(f"[leakage] universe {rep['n_goals']} goals (sha {rep['goal_universe']['sha256'][:12]}); fit list "
          f"{rep['harmful_fit_list']['n']} rows (sha {str(rep['harmful_fit_list']['sha256'])[:12]}); "
          f"exact {len(rep['exact'])} near {len(rep['near'])} total {len(rep['total'])} frac {rep['frac']:.4f}; "
          f"normalise_prompt {rep['normalise_prompt_origin']} sha {rep['normalise_prompt_source_sha256'][:12]}", flush=True)
    if args.frozen:
        fp = Path(args.frozen).expanduser()
        if not fp.is_file():
            raise Refusal(f"frozen artifact {fp} not found; pass --frozen '' only for a deliberate NEW freeze")
        frozen = json.loads(fp.read_text())
        problems = diff_against_frozen(rep, frozen)
        if problems:
            print("[leakage] DOES NOT REPRODUCE the frozen artifact " + str(fp) + ":", flush=True)
            for p_ in problems:
                print("   " + p_, flush=True)
            print("[leakage] STOP: a diff is a finding, not a restatement; nothing written (plan §5.4 item 4)", flush=True)
            return 2
        print(f"[leakage] reproduces the frozen artifact {fp} id-for-id (frozen sha256 {sha256_file(fp)[:12]})", flush=True)
    if args.check:
        return 0
    _write_json(Path(args.out).expanduser(), rep)
    print(f"[leakage] wrote {args.out} (payload sha {rep['sha256'][:12]})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
