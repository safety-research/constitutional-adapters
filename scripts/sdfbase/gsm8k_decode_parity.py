#!/usr/bin/env python
"""Decode-path parity attestation for ``base_gsm8k.py``: does the CACHED
decode reproduce a cell that was published under the CACHE-FREE decode,
row for row?

Input: the published cell's per-question file (the label's ``none``, written
before the fix — no ``runtime`` block on its aggregate) and the same cell
re-scored under the fixed path on a scratch label. Both must be pairable
(``base_gsm8k.PAIR_KEYS``), on the same model / revision / config / batch
size / dtype / attention backend, the same cell and object. Output: one JSON
attestation (``kind: gsm8k_decode_parity_attestation``) carrying

  * ``shared_identity`` — the identity fields the two runs share (what a
    later invocation must match: ``base_gsm8k.ATTEST_KEYS`` + cell/object);
  * ``published`` / ``new`` — each file, its sha256, its OUTCOME digest
    (``rows_outcome_digest``: qid + extracted + correct + stop_reason), the
    aggregate's ``runtime`` block when present (the cached run's), accuracy,
    non-terminating count, mean new tokens;
  * row-level agreement: identical response text / extracted answer /
    correct flag / stop reason / all three outcomes, the McNemar table with
    its exact two-sided p, the accuracy delta, where differing responses
    first diverge (characters), and every row whose extracted answer or
    stop reason differs (qidx, gold, both values);
  * ``verdict.outcomes_identical`` — True iff extracted, correct and
    stop_reason agree on EVERY row. That, and only that, lets
    ``base_gsm8k.require_decode_parity`` admit cached cells into a label
    that holds cache-free ones. Response text is reported, never required:
    the two bf16 kernel paths legitimately differ in tokens on some rows.

  python scripts/sdfbase/gsm8k_decode_parity.py \\
      --published <r2>/gsm8k_r2_none.per_question.jsonl \\
      --new <parity_x1>/gsm8k_parity_x1_none.per_question.jsonl \\
      --out decode_parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from math import comb
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import base_gsm8k as bg  # noqa: E402

#: Beyond ATTEST_KEYS (which a later run must match), the two files must
#: also be the SAME cell on the SAME object at the SAME scale.
SAME_RUN_KEYS = (*bg.ATTEST_KEYS, "cell", "object_sha256", "scale")


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant counts: the binomial
    (b + c, 1/2) tail at min(b, c), doubled and capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def first_divergence(a: str, b: str) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            return i
        i += 1
    return i


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _aggregate_next_to(pq_path: Path) -> dict | None:
    name = pq_path.name
    if not name.endswith(".per_question.jsonl"):
        return None
    agg = pq_path.with_name(name[:-len(".per_question.jsonl")] + ".json")
    if not agg.exists():
        return None
    return json.loads(agg.read_text(encoding="utf-8"))


def _side(path: Path, identity: dict, rows: list[dict]) -> dict:
    n = len(rows)
    agg = _aggregate_next_to(path) or {}
    return {
        "file": str(path), "per_question_sha256": _sha256(path),
        "run_label": identity.get("run_label"),
        "rows_digest": bg.rows_outcome_digest(rows),
        "runtime": agg.get("runtime"),
        "acc": sum(bool(r["correct"]) for r in rows) / n,
        "non_terminating": sum(not r["terminated"] for r in rows),
        "mean_new_tokens": sum(r["n_new_tokens"] for r in rows) / n,
    }


def compare(published: Path, new: Path) -> dict:
    ida, rows_a = bg.read_per_question(Path(published))
    idb, rows_b = bg.read_per_question(Path(new))
    bg.assert_pairable(ida, idb)
    for k in SAME_RUN_KEYS:
        if ida.get(k) != idb.get(k):
            raise SystemExit(f"[decode-parity] not the same run: {k} differs "
                             f"({ida.get(k)!r} vs {idb.get(k)!r})")
    by_b = {r["qid"]: r for r in rows_b}
    if set(by_b) != {r["qid"] for r in rows_a}:
        raise SystemExit("[decode-parity] qid sets differ")
    n = len(rows_a)
    same = {"response": 0, "extracted": 0, "correct": 0, "stop_reason": 0,
            "outcomes": 0}
    diverge_at: list[int] = []
    ext_diff: list[dict] = []
    stop_diff: list[dict] = []
    for ra in rows_a:
        rb = by_b[ra["qid"]]
        if ra["response"] == rb["response"]:
            same["response"] += 1
        else:
            diverge_at.append(first_divergence(ra["response"], rb["response"]))
        eq_ext = ra["extracted"] == rb["extracted"]
        eq_cor = bool(ra["correct"]) == bool(rb["correct"])
        eq_stop = ra.get("stop_reason") == rb.get("stop_reason")
        same["extracted"] += eq_ext
        same["correct"] += eq_cor
        same["stop_reason"] += eq_stop
        same["outcomes"] += eq_ext and eq_cor and eq_stop
        if not eq_ext:
            ext_diff.append({"qidx": ra["qidx"], "qid": ra["qid"],
                             "gold": ra["gold"],
                             "published": ra["extracted"], "new": rb["extracted"],
                             "published_correct": ra["correct"],
                             "new_correct": rb["correct"]})
        if not eq_stop:
            stop_diff.append({"qidx": ra["qidx"], "qid": ra["qid"],
                              "published": ra.get("stop_reason"),
                              "new": rb.get("stop_reason"),
                              "published_n_new": ra["n_new_tokens"],
                              "new_n_new": rb["n_new_tokens"]})
    t = bg.mcnemar_table(rows_a, rows_b)
    a, b = _side(Path(published), ida, rows_a), _side(Path(new), idb, rows_b)
    # The two sides must be the two IMPLEMENTATIONS, not two scratch files
    # that happen to agree: --published is the cache-free
    # legacy decode (no runtime block on its aggregate, or kv_cache False),
    # --new the cached re-score (an aggregate next to it with
    # runtime.kv_cache True).
    if a["runtime"] is not None and a["runtime"].get("kv_cache") is not False:
        raise SystemExit(
            "[decode-parity] --published must be the CACHE-FREE side: its "
            f"aggregate records runtime {a['runtime']} (decoded with the "
            "cache) — this comparison would never exercise the legacy path")
    if not b["runtime"] or not b["runtime"].get("kv_cache"):
        raise SystemExit(
            "[decode-parity] --new must be the CACHED re-score: its "
            "aggregate must sit next to the per-question file and record "
            f"runtime.kv_cache true (got {b['runtime']})")
    return {
        "kind": bg.ATTESTATION_KIND,
        "shared_identity": {k: ida.get(k) for k in SAME_RUN_KEYS},
        "published": a, "new": b, "n": n,
        "identical_response_text": same["response"],
        "identical_extracted_answer": same["extracted"],
        "identical_correct_flag": same["correct"],
        "identical_stop_reason": same["stop_reason"],
        "identical_outcomes": same["outcomes"],
        "acc_delta_pp": round(100 * (b["acc"] - a["acc"]), 3),
        "mcnemar": {**t, "p_exact_two_sided": mcnemar_exact_p(t["a_only"],
                                                              t["b_only"])},
        "text_divergence_chars": (
            {"n_rows": len(diverge_at), "min": min(diverge_at),
             "median": sorted(diverge_at)[len(diverge_at) // 2],
             "max": max(diverge_at)} if diverge_at else {"n_rows": 0}),
        "extracted_differs": ext_diff,
        "stop_reason_differs": stop_diff,
        "verdict": {"outcomes_identical": same["outcomes"] == n,
                    "outcome_keys": list(bg.OUTCOME_KEYS)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--published", required=True,
                    help="per-question file of the cell as published under "
                         "the label (the cache-free decode)")
    ap.add_argument("--new", required=True,
                    help="per-question file of the same cell re-scored under "
                         "the fixed (cached) decode on a scratch label")
    ap.add_argument("--out", default=None, help="attestation JSON path")
    args = ap.parse_args(argv)
    out = compare(Path(args.published), Path(args.new))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("extracted_differs", "stop_reason_differs")},
                     indent=1))
    n = out["n"]
    print(f"\nPARITY: outcomes identical on {out['identical_outcomes']}/{n} "
          f"rows (extracted {out['identical_extracted_answer']}, correct "
          f"{out['identical_correct_flag']}, stop_reason "
          f"{out['identical_stop_reason']}); byte-identical responses "
          f"{out['identical_response_text']}/{n}; acc "
          f"{out['published']['acc']:.3f} -> {out['new']['acc']:.3f} "
          f"({out['acc_delta_pp']:+.2f} pp, McNemar p="
          f"{out['mcnemar']['p_exact_two_sided']:.3g}, discordant "
          f"{out['mcnemar']['a_only']}+{out['mcnemar']['b_only']}); "
          f"non-terminating {out['published']['non_terminating']} -> "
          f"{out['new']['non_terminating']}; verdict outcomes_identical="
          f"{out['verdict']['outcomes_identical']}")
    for r in out["extracted_differs"]:
        print(f"  extracted differs: {r}")
    for r in out["stop_reason_differs"]:
        print(f"  stop reason differs: {r}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n",
                                  encoding="utf-8")
        print(f"[decode-parity] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
