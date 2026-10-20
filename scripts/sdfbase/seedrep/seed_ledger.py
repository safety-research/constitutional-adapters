#!/usr/bin/env python
"""The Qwen base seed-replicate lane: the SEED LEDGER — one row per (view, eval_id, cell, cap, trial) of the lane's
stores, each carrying the leg's ``seed_rule``, its ``seed_base``, the per-row seed FIELD, the realised decode-seed policy id
and the engine-resolved sampler record — with every seed a store claims RECOMPUTED from the leg's own derivation (never
trusted from the row) and the row REFUSED when it does not reproduce.

Input: a store INDEX the orchestrator / coverage.py writes — JSON list or JSONL, one entry per
``{view, eval_id, cell, cap, trial, store_dir[, resolved_sampler][, anchor_sampler_verdict]}`` (trial null on a leg without
trials). Output: ``RS/reseed/seed_ledger.jsonl`` (one JSON row per entry, in index order) and a one-line summary.

What each leg's reader checks (eval_seed.LEG_SEED_RULES names the rule; the derivations are the instruments' own):
  static families      responses.jsonl rows: sampling_seed == static_common.static_sampling_seed(base, trial, family, goal_id, sample_idx);
                       decode.seed_base == base, decode.trial == trial
  over-refusal         gens_<control>.meta.jsonl rows: sampling_seed == sampled_controls.sampling_seed(base, trial, control, item_id)
  mmlu                 instrument_frozen: request_evidence.jsonl records carry seed == 0 — if the records carry NO seed key the row
                       flips to none_sent FROM EVIDENCE (§13: the mechanism text is corrected from evidence, never assumed)
  gsm8k                request_evidence.jsonl task records: seed == contracts.inspect_replicate_seed(item_id, sample_idx, base)
  bon                  bon_responses.jsonl rows: seed_rule bon_seed_v1+decode:replicate_v1, decode_seed_policy replicate_v1,
                       decode_seed_base == base, augment_seed == bon_seed(...), sampling_seed == bon_decode_seed_v1(..., base);
                       bon_served_run.json's policy / base / preset when present
  multi-turn / petri / misalignment   none_sent: any request-evidence record carrying a non-null seed REFUSES
The engine-resolved sampler row (scripts/sdfbase/record_resolved_sampler.py's resolved_sampler.json — the entry's path, else the
store dir's, else a parent's) is recorded as {path, verdict, n}; ``anchor_sampler_verdict`` is the preflight's (W6b) when the
index carries it, null otherwise — this ledger never invents one.

Exit 0 when every row is `ok`; 1 when any row is REFUSED (a seed that does not reproduce, a wrong base or rule); 2 when a row
lacks evidence it needs (no rows / no request evidence / no sampler record) unless --allow-missing-evidence.

  python scripts/sdfbase/seedrep/seed_ledger.py --index RS/reseed/store_index.json --out RS/reseed/seed_ledger.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[3]
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as es  # noqa: E402

STATUS_OK = "ok"
STATUS_REFUSED = "refused"
STATUS_MISSING = "missing_evidence"
RESOLVED_SAMPLER_FILENAME = "resolved_sampler.json"
REQUEST_EVIDENCE_FILENAME = "request_evidence.jsonl"
INDEX_KEYS = ("view", "eval_id", "cell", "cap", "trial", "store_dir")
#: decisions O12a: the ONE unsteered measurement of the lane — the GSM8K paired-200 matrix's natively served `none` anchor row per view
#: (main's reader refuses the lens-era published card against a native cell by name, O106, and keys the anchor per view); its ledger
#: row draws under the view's base like every GSM8K row and is labelled `role: native_anchor` (every steered row `role: steered`)
NATIVE_ANCHOR_CELL = "none"
NATIVE_ANCHOR_EVAL = "gsm8k"
ROLE_STEERED, ROLE_NATIVE_ANCHOR = "steered", "native_anchor"
CELLS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce",
         "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl_distilled")


class Refusal(Exception):
    pass


def _jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue          # a torn final line (a SIGKILL mid-append) is skipped exactly as every resume does


def load_index(path: Path) -> list:
    text = Path(path).read_text(encoding="utf-8")
    entries = json.loads(text) if text.lstrip().startswith("[") else list(_jsonl(Path(path)))
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{path}: the store index is a non-empty JSON list (or JSONL) of entries")
    for i, e in enumerate(entries):
        missing = [k for k in INDEX_KEYS if k not in e]
        if missing:
            raise SystemExit(f"{path}: entry {i} lacks {missing} (every entry names {list(INDEX_KEYS)})")
        if not es.is_seedrep_view(e["view"]):
            raise SystemExit(f"{path}: entry {i}: view {e['view']!r} is not a seed-replicate view")
        es.leg_seed_rule(e["eval_id"])
        if e["cell"] not in CELLS and not (e["cell"] == NATIVE_ANCHOR_CELL and e["eval_id"] == NATIVE_ANCHOR_EVAL):
            raise SystemExit(f"{path}: entry {i}: cell {e['cell']!r} is not one of the lane's six cells (never `none` — except the GSM8K paired-200 "
                             f"NATIVE ANCHOR row of a view, decisions O12a — never a prompted baseline)")
        if isinstance(e["cap"], bool) or not isinstance(e["cap"], int) or e["cap"] < 1:
            raise SystemExit(f"{path}: entry {i}: cap must be a positive int, got {e['cap']!r}")
        if e["trial"] is not None and (isinstance(e["trial"], bool) or not isinstance(e["trial"], int) or e["trial"] < 0):
            raise SystemExit(f"{path}: entry {i}: trial must be a non-negative int or null, got {e['trial']!r}")
    return entries


def _find_resolved_sampler(entry: dict, store: Path) -> Optional[dict]:
    cands = []
    if entry.get("resolved_sampler"):
        cands.append(Path(entry["resolved_sampler"]))
    cands.append(store / RESOLVED_SAMPLER_FILENAME)
    cands += [p / RESOLVED_SAMPLER_FILENAME for p in list(store.parents)[:3]]
    for c in cands:
        if c.is_file():
            try:
                rec = json.loads(c.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise Refusal(f"{c}: unreadable resolved_sampler.json ({e})")
            return {"path": str(c), "verdict": rec.get("verdict"), "n": rec.get("n"), "reference_source": rec.get("reference_source")}
    return None


def _request_evidence(store: Path) -> Optional[list]:
    files = sorted(store.rglob(REQUEST_EVIDENCE_FILENAME)) + sorted(p for p in store.rglob(f"*.{REQUEST_EVIDENCE_FILENAME}"))
    if not files:
        return None
    out = []
    for f in files:
        out.extend(_jsonl(f))
    return out


def _read_static(store: Path, *, base: int, trial: int, family: str) -> dict:
    from sisteer.eval.attacks.static_common import static_sampling_seed
    files = sorted(store.rglob("responses.jsonl"))
    if not files:
        raise Refusal(f"{store}: no responses.jsonl under the store")
    n = 0
    for f in files:
        for row in _jsonl(f):
            dec = row.get("decode") or {}
            if int(dec.get("seed_base", -1)) != base or int(dec.get("trial", -1)) != trial:
                raise Refusal(f"{f}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r}) decode block records seed_base "
                              f"{dec.get('seed_base')!r} / trial {dec.get('trial')!r}, this view draws under {base} / {trial}")
            want = static_sampling_seed(base, trial, family, row["goal_id"], int(row["sample_idx"]))
            if row.get("sampling_seed") != want:
                raise Refusal(f"{f}: row (goal {row['goal_id']!r}, sample {row['sample_idx']!r}) records sampling_seed {row.get('sampling_seed')!r} "
                              f"but static_sampling_seed({base}, {trial}, {family}, goal, sample) = {want}")
            n += 1
    return {"n_rows_checked": n, "seeds_reproduce": True, "seed_field": "sampling_seed"}


def _read_controls(store: Path, *, base: int, trial: int, control: str) -> dict:
    from sisteer.eval.sampled_controls import META_SUFFIX, sampling_seed
    files = sorted(store.rglob(f"gens_{control}{META_SUFFIX}")) or sorted(store.rglob(f"*{META_SUFFIX}"))
    if not files:
        raise Refusal(f"{store}: no gens_{control}{META_SUFFIX} under the store")
    n = 0
    for f in files:
        for row in _jsonl(f):
            if "sampling_seed" not in row:
                continue          # a greedy / copied meta row without a seed is another instrument's; the sampled rows carry one
            want = sampling_seed(base, trial, control, str(row["item_id"]))
            if int(row["sampling_seed"]) != want:
                raise Refusal(f"{f}: item {row['item_id']!r} records sampling_seed {row['sampling_seed']} but "
                              f"sampling_seed({base}, {trial}, {control}, item) = {want}")
            n += 1
    if n == 0:
        raise Refusal(f"{store}: the meta rows carry no sampling_seed — not a sampled store")
    return {"n_rows_checked": n, "seeds_reproduce": True, "seed_field": "sampling_seed"}


def _read_bon(store: Path, *, base: int) -> dict:
    from sisteer.eval.attacks.bon import (DECODE_SEED_POLICY_REPLICATE, REPLICATE_ROW_KEYS, bon_decode_seed_v1, bon_seed,
                                          seed_rule_for_policy)
    files = sorted(store.rglob("bon_responses.jsonl"))
    if not files:
        raise Refusal(f"{store}: no bon_responses.jsonl under the store")
    want_rule = seed_rule_for_policy(DECODE_SEED_POLICY_REPLICATE)
    n = 0
    for f in files:
        for row in _jsonl(f):
            gid, idx, trial = row["goal_id"], int(row["sample_idx"]), int(row.get("trial", 0))
            if row.get("seed_rule") != want_rule:
                raise Refusal(f"{f}: row ({gid}, {idx}, {trial}) names seed_rule {row.get('seed_rule')!r}, the lane draws under {want_rule!r}")
            missing = [k for k in REPLICATE_ROW_KEYS if k not in row]
            if missing:
                raise Refusal(f"{f}: row ({gid}, {idx}, {trial}) lacks {missing}")
            if row["decode_seed_policy"] != DECODE_SEED_POLICY_REPLICATE or int(row["decode_seed_base"]) != base:
                raise Refusal(f"{f}: row ({gid}, {idx}, {trial}) records policy {row['decode_seed_policy']!r} / base {row['decode_seed_base']!r}, "
                              f"this view draws under {DECODE_SEED_POLICY_REPLICATE} / {base}")
            if int(row["augment_seed"]) != bon_seed(gid, idx, trial):
                raise Refusal(f"{f}: row ({gid}, {idx}, {trial}) augment_seed {row['augment_seed']} != bon_seed = {bon_seed(gid, idx, trial)} (the variant identity)")
            want = bon_decode_seed_v1(gid, idx, trial, base)
            if int(row["sampling_seed"]) != want:
                raise Refusal(f"{f}: row ({gid}, {idx}, {trial}) sampling_seed {row['sampling_seed']} != bon_decode_seed_v1 at base {base} = {want}")
            n += 1
    out = {"n_rows_checked": n, "seeds_reproduce": True, "seed_field": "sampling_seed"}
    run = sorted(store.rglob("bon_served_run.json"))
    if run:
        rec = json.loads(run[0].read_text(encoding="utf-8"))
        if rec.get("decode_seed_policy") != DECODE_SEED_POLICY_REPLICATE or rec.get("decode_seed_base") != base:
            raise Refusal(f"{run[0]}: records decode_seed_policy {rec.get('decode_seed_policy')!r} / decode_seed_base {rec.get('decode_seed_base')!r}, "
                          f"this view draws under {DECODE_SEED_POLICY_REPLICATE} / {base}")
        out["bon_decode_preset"] = rec.get("bon_decode_preset")
        out["run_record"] = str(run[0])
    return out


def _read_gsm8k(store: Path, *, base: int) -> dict:
    from sisteer.capeval.tinycap import contracts as C
    recs = _request_evidence(store)
    if recs is None:
        raise Refusal(f"{store}: no {REQUEST_EVIDENCE_FILENAME} under the store (the GSM8K seeds are proven from the wire record)")
    task = [r for r in recs if r.get("kind", "task") == "task" and r.get("item_id") is not None]
    if not task:
        raise Refusal(f"{store}: the request evidence carries no task record with an item_id")
    for r in task:
        want = C.inspect_replicate_seed(str(r["item_id"]), int(r.get("sample_idx") or 0), base)
        if r.get("seed") != want:
            raise Refusal(f"{store}: request for item {r['item_id']!r} sample {r.get('sample_idx')!r} carries seed {r.get('seed')!r} but "
                          f"inspect_replicate_seed(item, sample, {base}) = {want}")
    return {"n_rows_checked": len(task), "seeds_reproduce": True, "seed_field": "seed"}


def _read_mmlu(store: Path) -> dict:
    recs = _request_evidence(store)
    if recs is None:
        return {"n_rows_checked": 0, "seeds_reproduce": None, "seed_field": "seed", "evidence": None}
    task = [r for r in recs if r.get("kind", "task") == "task"]
    if not task:
        return {"n_rows_checked": 0, "seeds_reproduce": None, "seed_field": "seed", "evidence": None}
    with_seed = [r for r in task if "seed" in r and r["seed"] is not None]
    if not with_seed:
        # §13: the request evidence shows NO seed key -> the leg is none_sent, corrected FROM EVIDENCE
        return {"n_rows_checked": len(task), "seeds_reproduce": None, "seed_field": None, "evidence": "no seed key on the wire",
                "seed_rule_override": es.RULE_NONE_SENT}
    bad = [r for r in with_seed if int(r["seed"]) != es.MMLU_FROZEN_SEED]
    if bad or len(with_seed) != len(task):
        raise Refusal(f"{store}: MMLU request evidence: {len(bad)} record(s) carry a seed other than {es.MMLU_FROZEN_SEED}, "
                      f"{len(task) - len(with_seed)} carry none — a mixed wire is not the frozen instrument")
    return {"n_rows_checked": len(task), "seeds_reproduce": True, "seed_field": "seed", "evidence": f"seed {es.MMLU_FROZEN_SEED} on every task request",
            "frozen_seed": es.MMLU_FROZEN_SEED}


def _read_none_sent(store: Path) -> dict:
    recs = _request_evidence(store)
    if recs is None:
        return {"n_rows_checked": 0, "seeds_reproduce": None, "seed_field": None, "evidence": None}
    seeded = [r for r in recs if r.get("seed") is not None]
    if seeded:
        raise Refusal(f"{store}: a none_sent leg's request evidence carries a seed on {len(seeded)} record(s) — the instrument sent one")
    return {"n_rows_checked": len(recs), "seeds_reproduce": None, "seed_field": None, "evidence": "no seed on any recorded request"}


def ledger_row(entry: dict) -> dict:
    view, eval_id, cell = entry["view"], entry["eval_id"], entry["cell"]
    s = es.seed_of_view(view)
    leg = es.leg_seed_rule(eval_id)
    base = es.leg_seed_base(eval_id, s)
    store = Path(entry["store_dir"])
    row = {"view": view, "tag": es.tag_of(s), "seed": s, "eval_id": eval_id, "cell": cell, "cap": int(entry["cap"]), "trial": entry["trial"],
           "role": (ROLE_NATIVE_ANCHOR if cell == NATIVE_ANCHOR_CELL else ROLE_STEERED),
           "seed_rule": leg.seed_rule, "seed_base": base, "seed_field": leg.seed_field, "decode_seed_policy": leg.decode_seed_policy,
           "seed_mechanism": leg.mechanism, "store_dir": str(store), "n_rows_checked": 0, "seeds_reproduce": None,
           "resolved_sampler": None, "anchor_sampler_verdict": entry.get("anchor_sampler_verdict"), "status": STATUS_OK, "problems": []}
    try:
        if not store.is_dir():
            raise Refusal(f"{store}: no such store dir")
        trial = entry["trial"]
        if leg.seed_rule == es.RULE_REPLICATE and eval_id in ("pasttense", "codeattack", "renellm", "grok420", "xstest", "or_bench_hard_1k") and trial is None:
            raise Refusal(f"{eval_id}: a sampled leg's entry names its trial")
        if eval_id in ("pasttense", "codeattack", "renellm", "grok420"):
            row.update(_read_static(store, base=base, trial=int(trial), family=eval_id))
        elif eval_id in ("xstest", "or_bench_hard_1k"):
            row.update(_read_controls(store, base=base, trial=int(trial), control=eval_id))
        elif eval_id == "bon":
            row.update(_read_bon(store, base=base))
        elif eval_id == "gsm8k":
            row.update(_read_gsm8k(store, base=base))
        elif eval_id == "mmlu":
            row.update(_read_mmlu(store))
            if row.pop("seed_rule_override", None) == es.RULE_NONE_SENT:
                row["seed_rule"] = es.RULE_NONE_SENT
                row["seed_mechanism"] = "request evidence shows no seed key on the letter requests — corrected from evidence (§13)"
        else:
            row.update(_read_none_sent(store))
        if row.get("n_rows_checked", 0) == 0 and leg.seed_rule == es.RULE_REPLICATE:
            raise Refusal(f"{store}: no row checked on a replicate_seed_base leg")
        if row.get("n_rows_checked", 0) == 0:
            row["status"] = STATUS_MISSING
            row["problems"].append("no request evidence under the store — the leg's wire facts are unproven")
        row["resolved_sampler"] = _find_resolved_sampler(entry, store)
        if row["resolved_sampler"] is None:
            row["status"] = STATUS_MISSING
            row["problems"].append(f"no {RESOLVED_SAMPLER_FILENAME} for the store (record_resolved_sampler.py on the standup's server log)")
    except Refusal as e:
        row["status"] = STATUS_REFUSED
        row["problems"].append(str(e))
    except (KeyError, TypeError, ValueError) as e:
        row["status"] = STATUS_REFUSED
        row["problems"].append(f"unreadable store rows: {e!r}")
    return row


def build(entries: list) -> list:
    return [ledger_row(e) for e in entries]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--index", required=True, help="the store index (JSON list or JSONL) — one entry per (view, eval_id, cell, cap, trial)")
    ap.add_argument("--out", required=True, help="the ledger JSONL to write (RS/reseed/seed_ledger.jsonl)")
    ap.add_argument("--allow-missing-evidence", action="store_true",
                    help="exit 0 with rows marked missing_evidence (a standup that logged no requests, a sampler record not yet written); "
                         "a REFUSED row still exits 1")
    a = ap.parse_args(argv)
    rows = build(load_index(Path(a.index)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=False) + "\n" for r in rows), encoding="utf-8")
    n_ref = sum(r["status"] == STATUS_REFUSED for r in rows)
    n_miss = sum(r["status"] == STATUS_MISSING for r in rows)
    print(f"seed_ledger: {len(rows)} rows -> {out}: {len(rows) - n_ref - n_miss} ok, {n_miss} missing_evidence, {n_ref} refused")
    for r in rows:
        if r["status"] != STATUS_OK:
            print(f"  {r['status'].upper()} {r['view']} {r['eval_id']} {r['cell']} cap={r['cap']} trial={r['trial']}: {'; '.join(r['problems'])}", file=sys.stderr)
    if n_ref:
        return 1
    if n_miss and not a.allow_missing_evidence:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
