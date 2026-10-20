#!/usr/bin/env python
"""The campaign's degeneracy UNION classifier as a NON-DECIDING column beside a GSM8K paired-200 cell's cap ladder.

WHY. The cap ladder (``sisteer.capeval.tiers.cap_ladder``) decides a cell's rung eligibility with its own two-measure rule
(4-gram ``rep_ratio > 0.5`` on the whole text OR on the last 200 whitespace words: such an at-cap draw is ``non_terminating``,
never redrawn; the rest are ``truncated`` and eligible). The paper's other degeneracy numbers (best-of-N, the multi-turn
degenerate-win share) come from the campaign's ONE union classifier (``sisteer.eval.degeneracy_union``: five detectors).
The two are TWO INSTRUMENTS, never one series. This tool runs the union over the SAME at-cap draws the ladder
classified and files, per item, the ladder's kind, the strict-periodic flag the ladder record carries beside its rule,
and the union's verdict + reason — plus per-cell 2x2 agreement tables and the disagreeing item ids. It changes nothing:
eligibility stays the ladder's; the output is a new file BESIDE the cell's pulled store.

WHAT IS COMPARED. The union's fifth detector (``at_cap``) fires on every at-cap draw by construction, so the union's
combined ``degenerate`` is trivially true here. The agreement tables therefore pair the ladder's kind against the union's
CONTENT detectors (``classifier`` | ``rep_ratio_loop`` | ``zlib_collapse`` | ``contentless``; ``union_content``) and,
separately, against ``classifier`` alone (``sisteer.eval.degeneracy.classify`` — the tail_loop / line_loop rule).
Every detector's flag is filed per item so any other pairing can be read off the record.

TEXT IDENTITY. The at-cap texts are read from the cell's Inspect ``.eval`` log (``sample.output.completion``) and every one
is held to the ladder record's ``text_sha256``; a mismatch refuses the cell (the union would have classified a different
text than the ladder did). A rung store (``__rung16384``) carries no ``cap_ladder.json``; there the ladder's rule and the
periodic block are RE-DERIVED through the modules' own functions over the log text and the source is filed as such.

RUNS in a venv with ``inspect_ai`` and ``transformers`` (the lane venv; the login node's python cannot open the zstd zip),
with ``PYTHONPATH=<repo>/src`` so ``sisteer`` resolves to the repo's validated bytes. No prompt or response text is stored;
per-item records carry ids, hashes, counts and flags only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.eval import degeneracy_union as DU

SCHEMA = "degeneracy_union_column/1"
OUT_NAME = "degeneracy_union_column.json"
#: Inspect spells the provider's at-cap stop as ``max_tokens``; the union's ``at_cap_finish_reason`` estimand is defined on the
#: OpenAI-shaped ``length``. The mapping is filed in the output; nothing else about the stop reason is interpreted.
FINISH_REASON_OF_STOP = {"max_tokens": "length", "stop": "stop"}
CONTENT_DETECTORS = ("classifier", "rep_ratio_loop", "zlib_collapse", "contentless")
LADDER_DEGENERATE_KIND = CL.KIND_NON_TERMINATING
LADDER_TRUNCATED_KINDS = (CL.KIND_TRUNCATED, CL.KIND_BOUNDARY)
SOURCE_FILED, SOURCE_REDERIVED = "cap_ladder.json (filed by the ladder)", "re-derived (no cap_ladder.json at this store)"


class Refuse(SystemExit):
    def __init__(self, why: str):
        print(f"[degeneracy-union-column] REFUSED: {why}", file=sys.stderr, flush=True)
        super().__init__(2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the store

def cell_dirs(out_root: Path) -> list[Path]:
    cells = out_root / "cells"
    return sorted(p for p in cells.iterdir() if p.is_dir()) if cells.is_dir() else []


def eval_logs(cell_dir: Path) -> list[Path]:
    return sorted((cell_dir / "private" / "inspect_logs").glob("*/*.eval"))


def read_eval_texts(path: Path) -> dict[str, dict[str, Any]]:
    """``{sample id: {text, stop_reason, output_tokens}}`` from one Inspect log (``sample.output.completion``; a sample id
    named twice refuses — the log must be one draw per item)."""
    from inspect_ai.log import read_eval_log  # the venv with inspect_ai; imported here so --help works without it

    log = read_eval_log(str(path))
    out: dict[str, dict[str, Any]] = {}
    for s in log.samples or []:
        sid = str(s.id)
        if sid in out:
            raise Refuse(f"{path}: sample {sid} appears twice (k = 1 expected)")
        choice = s.output.choices[0] if s.output.choices else None
        usage = s.output.usage
        # the EMITTED text as the ladder reads it (inspect_rung.emitted_text: the reasoning channel, then the content): a string
        # message is content alone; a parts list carries the reasoning channel as ContentReasoning parts beside the text parts
        raw = choice.message.content if choice is not None else ""
        if isinstance(raw, str):
            content, reasoning = raw, None
        else:
            content = "".join(getattr(p, "text", "") or "" for p in raw if getattr(p, "type", None) == "text")
            reasoning_parts = [getattr(p, "reasoning", "") or "" for p in raw if getattr(p, "type", None) == "reasoning"]
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
        out[sid] = {"text": IR.emitted_text(content, reasoning) or "", "stop_reason": (choice.stop_reason if choice is not None else None),
                    "output_tokens": (int(usage.output_tokens) if usage is not None and usage.output_tokens is not None else None)}
    return out


def manifest_of(cell_dir: Path) -> dict[str, Any]:
    p = cell_dir / "manifest.json"
    if not p.is_file():
        raise Refuse(f"{cell_dir}: no manifest.json")
    return read_json(p)


def load_tokenizer(model_id: str, revision: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)


# --------------------------------------------------------------------------- the ladder side

def ladder_facts(out_root: Path) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any] | None, Path | None]:
    """The filed ladder record's per-item facts keyed by item id, the record and its path — or ``(None, None, None)``
    when the store carries none (a rung store)."""
    p = out_root / CL.RECORD_NAME
    if not p.is_file():
        return None, None, None
    rec = read_json(p)
    facts = {str(f["item_id"]): f for f in rec.get("item_facts") or []}
    if len(facts) != len(rec.get("item_facts") or []):
        raise Refuse(f"{p}: item_facts names an item twice")
    return facts, rec, p


def rederived_fact(item_id: str, text: str, *, cap: int, at_cap: bool, tok) -> dict[str, Any]:
    """The ladder's own rule and the periodic block, RE-DERIVED over the log text through the modules' functions."""
    deg = CL.degeneracy(text)
    kind = (LADDER_DEGENERATE_KIND if deg["degenerate"] else CL.KIND_TRUNCATED) if at_cap else None
    return {"item_id": item_id, "at_cap": at_cap, "degenerate": bool(deg["degenerate"]), "rep_ratio": deg["rep_ratio"],
            "tail_rep_ratio": deg["tail_rep_ratio"], "kind": kind, "periodic_block": IR.periodic_block(text),
            "completion_tokens": min(CL.ntok(tok, text) + 1, cap), "source_cap": cap, "text_sha256": CL.text_sha256(text)}


# --------------------------------------------------------------------------- the tables

def two_by_two(items: Sequence[Mapping[str, Any]], *, union_key: str) -> dict[str, Any]:
    """Ladder kind (degenerate = non_terminating; truncated = truncated | boundary) x the union column ``union_key``."""
    cells = {"ladder_degenerate__union_degenerate": [], "ladder_degenerate__union_clean": [],
             "ladder_truncated__union_degenerate": [], "ladder_truncated__union_clean": []}
    for it in items:
        ladder = "ladder_degenerate" if it["ladder_kind"] == LADDER_DEGENERATE_KIND else "ladder_truncated"
        union = "union_degenerate" if it["union"][union_key] else "union_clean"
        cells[f"{ladder}__{union}"].append(it["item_id"])
    n = len(items)
    agree = len(cells["ladder_degenerate__union_degenerate"]) + len(cells["ladder_truncated__union_clean"])
    return {"union_column": union_key, "n_at_cap": n, "counts": {k: len(v) for k, v in cells.items()}, "ids": cells,
            "n_agree": agree, "agreement_share": (agree / n) if n else None,
            "disagreements": {"ladder_degenerate_union_clean": cells["ladder_degenerate__union_clean"],
                              "ladder_truncated_union_degenerate": cells["ladder_truncated__union_degenerate"]}}


# --------------------------------------------------------------------------- one cell

def classify_cell(out_root: Path, *, tokenizer_cache: dict[tuple[str, str | None], Any], no_retokenize: bool = False) -> dict[str, Any]:
    out_root = out_root.resolve()
    dirs = cell_dirs(out_root)
    if len(dirs) != 1:
        raise Refuse(f"{out_root}: expected exactly one cells/<cell id> dir, found {len(dirs)}")
    cell_dir = dirs[0]
    manifest = manifest_of(cell_dir)
    facts, record, record_path = ladder_facts(out_root)
    if facts is not None and record is not None:
        store_dir = str(record.get("store_dir") or "")
        if store_dir and Path(store_dir).name != cell_dir.name:
            raise Refuse(f"{record_path}: store_dir names {Path(store_dir).name}, the store's cell dir is {cell_dir.name}")
    logs = eval_logs(cell_dir)
    if not logs:
        raise Refuse(f"{cell_dir}: no private/inspect_logs/*/*.eval")
    # the store's cap: the ladder record's where filed, else the manifest's per-task cap (the rung store's 16,384)
    if record is not None:
        cap = int(record["cap"])
    else:
        by_task = manifest.get("on_cap_by_task") or {}
        cap = int(by_task.get("inspect/gsm8k_paired200") or by_task.get("gsm8k_paired200") or 0)
        if not cap:
            # the store's own task summary names the cap it was drawn at (the rung store's 16,384)
            summaries = sorted((cell_dir / "summary").glob("*.json")) if (cell_dir / "summary").is_dir() else []
            caps = {int(read_json(p).get("max_tokens") or 0) for p in summaries} - {0}
            if len(caps) != 1:
                raise Refuse(f"{cell_dir}: no cap_ladder.json, manifest.on_cap_by_task names no gsm8k_paired200 cap and the task summary "
                             f"names {sorted(caps) or 'no'} max_tokens")
            cap = caps.pop()
    model_id, revision = str(manifest.get("model_id")), (str(manifest["model_revision"]) if manifest.get("model_revision") else None)
    tok = None
    if not no_retokenize:
        key = (model_id, revision)
        if key not in tokenizer_cache:
            tokenizer_cache[key] = load_tokenizer(model_id, revision)
        tok = tokenizer_cache[key]
    # the ONE log whose samples are the ladder's items (hash-matched); several logs (retries) are disambiguated by that match
    chosen: tuple[Path, dict[str, dict[str, Any]]] | None = None
    match_report: list[dict[str, Any]] = []
    for lp in logs:
        texts = read_eval_texts(lp)
        if facts is not None:
            matched = sum(1 for iid, f in facts.items() if iid in texts and CL.text_sha256(texts[iid]["text"]) == f["text_sha256"])
            rep = {"log": str(lp), "n_samples": len(texts), "n_hash_matched": matched, "n_ladder_items": len(facts)}
            match_report.append(rep)
            if matched == len(facts) and len(texts) == len(facts):
                if chosen is not None:
                    raise Refuse(f"{cell_dir}: two logs match the ladder record whole ({chosen[0].name}, {lp.name})")
                chosen = (lp, texts)
        else:
            match_report.append({"log": str(lp), "n_samples": len(texts), "n_hash_matched": None, "n_ladder_items": None})
            if chosen is not None:
                raise Refuse(f"{cell_dir}: no ladder record to disambiguate {len(logs)} logs")
            chosen = (lp, texts)
    if chosen is None:
        raise Refuse(f"{cell_dir}: no .eval log whose samples hash-match the ladder record's items ({match_report})")
    log_path, texts = chosen
    # the at-cap items: the ladder's where filed (its own at_cap rule), else the log's stop reason at the cap
    if facts is not None:
        at_cap_ids = sorted(iid for iid, f in facts.items() if f["at_cap"])
        n_items = len(facts)
        fact_source = SOURCE_FILED
    else:
        at_cap_ids = sorted(iid for iid, t in texts.items() if t["stop_reason"] == "max_tokens")
        n_items = len(texts)
        fact_source = SOURCE_REDERIVED
    items: list[dict[str, Any]] = []
    rederived_disagreements: list[dict[str, Any]] = []
    for iid in at_cap_ids:
        t = texts[iid]
        text = t["text"]
        if facts is not None:
            fact = facts[iid]
        else:
            fact = rederived_fact(iid, text, cap=cap, at_cap=True, tok=tok if tok is not None else load_tokenizer(model_id, revision))
        # the ladder's rule re-derived over the SAME text as a consistency check on a filed record (never a replacement)
        re_deg = CL.degeneracy(text)
        if facts is not None and bool(re_deg["degenerate"]) != bool(fact["degenerate"]):
            rederived_disagreements.append({"item_id": iid, "filed": bool(fact["degenerate"]), "rederived": bool(re_deg["degenerate"]),
                                            "rep_ratio": re_deg["rep_ratio"], "tail_rep_ratio": re_deg["tail_rep_ratio"]})
        n_tokens = None if tok is None else int(DU.count_tokens(tok, [text])[0])
        finish = FINISH_REASON_OF_STOP.get(str(t["stop_reason"]), t["stop_reason"])
        u = DU.classify_response(text, cap=cap, n_tokens=(n_tokens if n_tokens is not None else fact.get("completion_tokens")),
                                 finish_reason=finish, at_cap=True)
        content_fired = [k for k in CONTENT_DETECTORS if u[k]]
        pb = fact.get("periodic_block") or {}
        items.append({
            "item_id": iid, "text_sha256": CL.text_sha256(text), "source_cap": int(fact.get("source_cap") or cap),
            "completion_tokens": fact.get("completion_tokens"), "stop_reason": t["stop_reason"],
            "ladder_kind": fact["kind"], "ladder_degenerate": bool(fact["degenerate"]),
            "ladder_rep_ratio": fact.get("rep_ratio"), "ladder_tail_rep_ratio": fact.get("tail_rep_ratio"),
            "strict_periodic": bool(pb.get("loop")) if pb else None,
            "periodic_block": {k: pb.get(k) for k in ("loop", "period", "repeats", "cover")} if pb else None,
            "union": {**{k: u[k] for k in DU.DETECTORS}, "degenerate": u["degenerate"], "reason": u["reason"],
                      "classifier_reason": u["classifier_reason"], "rep_ratio": u["rep_ratio"], "zlib_ratio": u["zlib_ratio"],
                      "alnum_count": u["alnum_count"], "n_tokens_retokenized": u["n_tokens"],
                      "at_cap_retokenized": u["at_cap_retokenized"], "at_cap_finish_reason": u["at_cap_finish_reason"],
                      "at_cap_field_disagrees": u["at_cap_field_disagrees"],
                      "union_content": bool(content_fired), "union_content_reason": "+".join(content_fired) if content_fired else None},
        })
    detector_counts = {k: sum(1 for it in items if it["union"][k]) for k in DU.DETECTORS}
    detector_counts["union_content"] = sum(1 for it in items if it["union"]["union_content"])
    detector_counts["union_any"] = sum(1 for it in items if it["union"]["degenerate"])
    n_deg = sum(1 for it in items if it["ladder_kind"] == LADDER_DEGENERATE_KIND)
    n_trunc = sum(1 for it in items if it["ladder_kind"] in LADDER_TRUNCATED_KINDS)
    n_periodic = sum(1 for it in items if it["strict_periodic"])
    tables = {"ladder_x_union_content": two_by_two(items, union_key="union_content"),
              "ladder_x_classifier": two_by_two(items, union_key="classifier"),
              "ladder_x_rep_ratio_loop": two_by_two(items, union_key="rep_ratio_loop")}
    union_files = {name: {"path": str(Path(DU.__file__).resolve().parents[0] / f"{name}.py"),
                          "sha256": sha256_file(Path(DU.__file__).resolve().parents[0] / f"{name}.py")}
                   for name in ("degeneracy_union", "degeneracy", "rep_ratio", "static_grading")}
    return {
        "schema": SCHEMA, "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deciding": False,
        "what": ("the campaign's degeneracy UNION classifier over the SAME at-cap draws the cap ladder classified — a non-deciding "
                 "column; eligibility stays the ladder's rule (D118); the two are two instruments, never one series (D117)"),
        "cell": {"out_root": str(out_root), "cell_dir": str(cell_dir), "cell_id": cell_dir.name, "model_id": model_id, "model_revision": revision,
                 "cap": cap, "is_rung": record is None, "n_items": n_items,
                 "producer_tree_sha": (manifest.get("producer") or {}).get("tree_sha") if isinstance(manifest.get("producer"), Mapping) else None},
        "sources": {"cap_ladder": ({"path": str(record_path), "sha256": sha256_file(record_path), "schema": record.get("schema"), "exit": record.get("exit")}
                                   if record_path is not None else None),
                    "ladder_facts": fact_source,
                    "eval_log": {"path": str(log_path), "sha256": sha256_file(log_path), "n_samples": len(texts)},
                    "eval_log_candidates": match_report,
                    "text_identity": ("sha256(sample.output.completion) == cap_ladder item_facts.text_sha256 on every ladder item" if facts is not None
                                      else "no ladder record: texts are the log's; text_sha256 filed per item"),
                    "union_module": {"definition_sha": DU.DEFINITION_SHA, "schema": DU.SCHEMA, "files": union_files},
                    "tokenizer": ({"model_id": model_id, "revision": revision, "local_files_only": True} if tok is not None
                                  else {"note": "--no-retokenize: n_tokens = the producer's completion_tokens (RECORDED, not the canonical estimand)"}),
                    "finish_reason_mapping": FINISH_REASON_OF_STOP},
        "rules": {"ladder_degenerate": CL.DEGENERATE_RULE, "ladder_trigger": CL.TRIGGER_RULE,
                  "ladder_periodic_block": (record.get("rules") or {}).get("periodic_block") if record else IR.periodic_block.__doc__,
                  "union_detectors": list(DU.DETECTORS), "union_content_detectors": list(CONTENT_DETECTORS),
                  "union_thresholds": {"rep_ratio_loop": DU.LOOP_REP_RATIO, "zlib_collapse": DU.ZLIB_COLLAPSE_RATIO,
                                       "contentless_alnum_min": DU.CONTENTLESS_ALNUM_MIN, "at_cap_retokenized_fraction": DU.AT_CAP_RETOKENIZED_FRACTION},
                  "note": ("the union's fifth detector (at_cap) fires on every at-cap draw by construction, so union.degenerate is true on every "
                           "row here; the agreement tables pair the ladder's kind with the CONTENT detectors and with classifier alone")},
        "counts": {"n_at_cap": len(items), "ladder_non_terminating": n_deg, "ladder_truncated_or_boundary": n_trunc,
                   "strict_periodic": n_periodic, "union_detectors": detector_counts,
                   "filed_record": {k: record.get(k) for k in ("n", "n_at_cap", "n_non_terminating", "n_boundary", "n_eligible", "exit")} if record else None,
                   "periodic_block_filed_n_loop": (record.get("periodic_block") or {}).get("n_loop") if record else None},
        "ladder_rule_rederived_disagreements": rederived_disagreements,
        "agreement": tables,
        "items": items,
    }


# --------------------------------------------------------------------------- the summary

def summary_lines(results: Sequence[tuple[Path, Mapping[str, Any]]]) -> list[str]:
    lines = ["# Degeneracy union vs the cap ladder — qwen-it GSM8K paired-200 cells (non-deciding)", "",
             "Ladder kind = the deciding rule (`non_terminating` = degenerate under rep_ratio > 0.5 on the text or its last 200 words; "
             "`truncated` = eligible). Union content = classifier | rep_ratio_loop | zlib_collapse | contentless; the union's at_cap "
             "detector fires on every row here by construction and is left out of the pairing. Strict periodic = the ladder record's "
             "`periodic_block.loop` (period 8-400 chars, >= 3 repeats, >= 60 % of the last 2,000 chars).", "",
             "| cell | cap | n at cap | ladder degenerate | of which strict periodic | ladder truncated | union content degenerate | classifier alone | "
             "agree (content) | agree (classifier) | ladder-deg & union-clean | ladder-trunc & union-deg |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for out_root, r in results:
        c, t = r["counts"], r["agreement"]
        tc, tk = t["ladder_x_union_content"], t["ladder_x_classifier"]
        name = "/".join(out_root.parts[-3:-1]) if r["cell"]["is_rung"] is False else "/".join(out_root.parts[-3:])
        lines.append(f"| {name} | {r['cell']['cap']} | {c['n_at_cap']} | {c['ladder_non_terminating']} | {c['strict_periodic']} | "
                     f"{c['ladder_truncated_or_boundary']} | {c['union_detectors']['union_content']} | {c['union_detectors']['classifier']} | "
                     f"{tc['n_agree']}/{tc['n_at_cap']} | {tk['n_agree']}/{tk['n_at_cap']} | "
                     f"{len(tc['disagreements']['ladder_degenerate_union_clean'])} | {len(tc['disagreements']['ladder_truncated_union_degenerate'])} |")
    lines += ["", "## Disagreements by item id (ladder kind vs union content)", ""]
    for out_root, r in results:
        tc = r["agreement"]["ladder_x_union_content"]
        d1, d2 = tc["disagreements"]["ladder_degenerate_union_clean"], tc["disagreements"]["ladder_truncated_union_degenerate"]
        if not d1 and not d2:
            continue
        lines.append(f"### {out_root}")
        by_id = {it["item_id"]: it for it in r["items"]}
        if d1:
            lines.append(f"- ladder DEGENERATE, union content CLEAN ({len(d1)}): " + ", ".join(
                f"{i} (rep {by_id[i]['ladder_rep_ratio']:.2f} / tail {by_id[i]['ladder_tail_rep_ratio']:.2f}; union rep {by_id[i]['union']['rep_ratio']:.2f}, "
                f"zlib {by_id[i]['union']['zlib_ratio']:.2f}, periodic {by_id[i]['strict_periodic']})" for i in d1))
        if d2:
            lines.append(f"- ladder TRUNCATED, union content DEGENERATE ({len(d2)}): " + ", ".join(
                f"{i} ({by_id[i]['union']['union_content_reason']}; ladder rep {by_id[i]['ladder_rep_ratio']:.2f} / tail {by_id[i]['ladder_tail_rep_ratio']:.2f})" for i in d2))
        lines.append("")
    lines += ["## Per-detector counts over the at-cap draws", ""]
    for out_root, r in results:
        lines.append(f"- {out_root}: " + ", ".join(f"{k} {v}" for k, v in r["counts"]["union_detectors"].items())
                     + (f"; ladder-rule re-derivation disagreements: {len(r['ladder_rule_rederived_disagreements'])}" if r["ladder_rule_rederived_disagreements"] else ""))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell-root", action="append", default=[], metavar="OUT_ROOT",
                    help="a pulled store's out root (…/<cell slug>/thinking_card[__rung16384]: cap_ladder.json beside cells/<cell id>/); repeatable")
    ap.add_argument("--out-name", default=OUT_NAME, help=f"the file written BESIDE each out root's cap_ladder.json (default {OUT_NAME}); refuses to overwrite")
    ap.add_argument("--summary", type=Path, default=None, help="a markdown summary over every cell (written once)")
    ap.add_argument("--no-retokenize", action="store_true", help="use the producer's completion_tokens for the union's at_cap detector (RECORDED; not canonical)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file (default: refuse)")
    a = ap.parse_args(argv)
    if not a.cell_root:
        ap.error("at least one --cell-root")
    tok_cache: dict[tuple[str, str | None], Any] = {}
    results: list[tuple[Path, dict[str, Any]]] = []
    for root in a.cell_root:
        out_root = Path(root).expanduser().resolve()
        target = out_root / a.out_name
        if target.exists() and not a.overwrite:
            raise Refuse(f"{target} exists (a new file beside the store, never an overwrite; --overwrite to replace)")
        r = classify_cell(out_root, tokenizer_cache=tok_cache, no_retokenize=a.no_retokenize)
        target.write_text(json.dumps(r, indent=1, sort_keys=False) + "\n", encoding="utf-8")
        c, tc = r["counts"], r["agreement"]["ladder_x_union_content"]
        print(f"[degeneracy-union-column] {out_root}: n_at_cap {c['n_at_cap']} ladder-degenerate {c['ladder_non_terminating']} "
              f"(strict periodic {c['strict_periodic']}) truncated {c['ladder_truncated_or_boundary']} | union content {c['union_detectors']['union_content']} "
              f"classifier {c['union_detectors']['classifier']} | agree {tc['n_agree']}/{tc['n_at_cap']} -> {target}")
        results.append((out_root, r))
    if a.summary is not None:
        a.summary.parent.mkdir(parents=True, exist_ok=True)
        a.summary.write_text("\n".join(summary_lines(results)) + "\n", encoding="utf-8")
        print(f"[degeneracy-union-column] summary -> {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
