#!/usr/bin/env python3
"""ifeval-canonical-battery — G-LENGTHS: the lengths-only PROMPT MEASUREMENT the three ``capability:ifeval_*`` geometry
entries are REGISTERED from. CPU only; no volume writes; the tokenizers read
OFFLINE from the local HF cache at the pinned revisions.

For every IFEval model key (gemma4e4b, qwen35_4b, nemotron3nano4b, llama8b) x prompt class (bare = the 541 stock renders;
prompted = the same with the safety SI folded into ONE leading system turn; constitution = the same with the SHORT
constitution SI, ``prompted__si_constitution``, ~6-6.7k tokens — the follow-up): the token count of every
rendered prompt under the thinking-OFF render of the PINNED tokenizer (``scripts/tinycap_oos_prompt_budget.py``'s
``prompt_token_count``, the same call the ``--task ifeval`` branch makes), its ``chat_template_hash``, and the nearest-rank
statistics. Then per cap of the ladder, TWO entries: ``capability:ifeval_<cap>`` with ``L = geometry.next_block(max prompt
over every model of the SHORT classes (bare, prompted) + cap)`` and ``capability:ifeval_longsi_<cap>`` with the max over
the constitution class — the MAX-prompt branch (``SIZED_BY_MAX_PROMPT_PLUS_MAX``, MATH-500's and the GSM8K rung's), so
no cap-bound draw is ever clamped by the window, and the short cells are never served at the constitution's window.

Writes ``configs/sdfbase/ifeval/geometry_measurement.json`` (no timestamp; ``--tree-sha`` recorded, so a regeneration
at the same tree is byte-identical — the test pins it) and, with ``--artifacts-dir``, one contract-exact
``oos_prompt_budget.json`` per (model, class, cap) OUTSIDE the repo (the lengths artifacts of the G-LENGTHS pass).
Lane 1 registers the entries from this file's ``entries`` block; nothing here types a window ahead of the measurement.

  PYTHONPATH=src python scripts/sdfbase/ifeval/measure_ifeval_geometry.py --offline [--tree-sha <40hex>] \\
      [--out configs/sdfbase/ifeval/geometry_measurement.json] [--artifacts-dir $GRADE/tinycap/ifeval/lengths]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.tiers import ifeval_messages as IFM  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import ifeval_contracts as IC  # noqa: E402
from sisteer.capeval.tinycap.completion import atomic_write_json  # noqa: E402

SCHEMA = "ifeval_geometry_measurement/1"
CLASS_BARE, CLASS_PROMPTED, CLASS_CONSTITUTION = "bare", "prompted", "constitution"
CLASSES = (CLASS_BARE, CLASS_PROMPTED, CLASS_CONSTITUTION)
#: the classes each entry family is sized over (contracts.IFEVAL_PROMPT_CLASS_*): short = bare + the safety SI
SHORT_CLASSES = (CLASS_BARE, CLASS_PROMPTED)
LONG_SI_CLASSES = (CLASS_CONSTITUTION,)
#: the prompted cell whose system turn each prompted class folds in (bare folds none)
SYSTEM_PROMPT_CELL_OF_CLASS = {CLASS_PROMPTED: IC.CELL_PROMPTED, CLASS_CONSTITUTION: IC.CELL_PROMPTED_CONSTITUTION}
OUT_DEFAULT = REPO / IC.IFEVAL_GEOMETRY_MEASUREMENT_REL


def _budget_module():
    spec = importlib.util.spec_from_file_location("tinycap_oos_prompt_budget", REPO / "scripts" / "tinycap_oos_prompt_budget.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tree_sha() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def measure(*, model_keys=IC.IFEVAL_MODELS, caps=IC.IFEVAL_CAP_LADDER, tree: Optional[str] = None,
            artifacts_dir: Optional[Path] = None, loader=None, renderer=None) -> Dict[str, Any]:
    """The measurement document. ``loader(model_key) -> (hf, rev, tok)`` and ``renderer(system_prompt) -> messages`` are
    seams for the tests (a synthetic tokenizer / renderer); production uses the budget CLI's pinned loader + the
    stock renderer."""
    B = _budget_module()
    from sisteer.models import chat_template_hash

    if loader is None:
        def loader(key):
            hf, rev, _, tok = B._load_ifeval_tokenizer(key, None)
            return hf, rev, tok
    if renderer is None:
        def renderer(system_prompt):
            return B.ifeval_class_messages(REPO, system_prompt=system_prompt)
    si = B.system_prompt_for_cell(IC.CELL_PROMPTED)
    si_constitution = B.system_prompt_for_cell(IC.CELL_PROMPTED_CONSTITUTION)
    if hashlib.sha256(si_constitution.encode("utf-8")).hexdigest() != C.IFEVAL_CONSTITUTION_SI_SHA256:
        raise SystemExit(f"the {IC.CELL_PROMPTED_CONSTITUTION} system text's sha256 is not the pinned "
                         f"{C.IFEVAL_CONSTITUTION_SI_SHA256} — the instruction text moved; re-pin deliberately")
    rendered = {CLASS_BARE: renderer(None), CLASS_PROMPTED: renderer(si), CLASS_CONSTITUTION: renderer(si_constitution)}
    for cls, msgs in rendered.items():
        if len(msgs) != IC.IFEVAL_N:
            raise SystemExit(f"{cls}: rendered {len(msgs)} prompts, not {IC.IFEVAL_N}")
    try:
        import transformers
        tf = transformers.__version__
    except ImportError:
        tf = None
    by_model: Dict[str, Dict[str, Any]] = {}
    for key in model_keys:
        hf, rev, tok = loader(key)
        entry: Dict[str, Any] = {"hf": hf, "revision": rev, "chat_template_hash": chat_template_hash(tok), "classes": {}}
        for cls in CLASSES:
            lengths = [B.prompt_token_count(tok, m, C.THINKING_OFF, content_format=C.prompt_budget_content_format(key))
                       for m in rendered[cls]]
            entry["classes"][cls] = {"n": len(lengths), "max": max(lengths), "p99": B.percentile(lengths, 0.99),
                                     "p50": B.percentile(lengths, 0.50), "mean": round(sum(lengths) / len(lengths), 2),
                                     "sum": sum(lengths), "lengths": lengths}
            if artifacts_dir is not None:
                for cap in caps:
                    doc = B.build_prompt_budget(rendered[cls], tok, arm=C.THINKING_OFF, model=hf, model_slug=key, revision=rev,
                                               content_format=C.prompt_budget_content_format(key),
                                                cap=int(cap), cap_task=IC.IFEVAL_TASK, tree_sha=tree,
                                                messages_source=IFM.messages_source(prompted=(cls == CLASS_PROMPTED)))
                    atomic_write_json(Path(artifacts_dir) / key / cls / f"off_{cap}" / C.OOS_PROMPT_BUDGET_FILENAME, doc)
        si_delta = [b - a for a, b in zip(entry["classes"][CLASS_BARE]["lengths"], entry["classes"][CLASS_PROMPTED]["lengths"])]
        entry["si_delta"] = {"min": min(si_delta), "max": max(si_delta)}
        con_delta = [b - a for a, b in zip(entry["classes"][CLASS_BARE]["lengths"], entry["classes"][CLASS_CONSTITUTION]["lengths"])]
        entry["constitution_si_delta"] = {"min": min(con_delta), "max": max(con_delta)}
        by_model[key] = entry

    def _stats(classes):
        mx = max(e["classes"][cls]["max"] for e in by_model.values() for cls in classes)
        p99 = B.percentile([x for e in by_model.values() for cls in classes for x in e["classes"][cls]["lengths"]], 0.99)
        am = max(((e["classes"][cls]["max"], key, cls) for key, e in by_model.items() for cls in classes))
        return mx, p99, {"model_key": am[1], "class": am[2]}

    def _entries(names_by_cap, mx, classes):
        out = {}
        for cap in caps:
            L = sg.next_block(mx + int(cap))
            out[str(cap)] = {"name": names_by_cap[int(cap)], "requested_max_tokens": int(cap), "classes": list(classes),
                             "max_prompt": mx, "max_model_len": L, "sized_by": sg.SIZED_BY_MAX_PROMPT_PLUS_MAX,
                             "block": sg.VLLM_BLOCK_SIZE, "headroom": L - mx - int(cap),
                             "starting_seqs_by_model": {k: _seqs_at(k, L) for k in model_keys}}
        return out
    max_prompt, p99_pooled, argmax = _stats(SHORT_CLASSES)
    max_prompt_longsi, p99_pooled_longsi, argmax_longsi = _stats(LONG_SI_CLASSES)
    entries = _entries(IC.IFEVAL_GEOMETRY_ENTRY_BY_CAP, max_prompt, SHORT_CLASSES)
    entries_longsi = _entries(IC.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI, max_prompt_longsi, LONG_SI_CLASSES)
    return {
        "schema": SCHEMA, "instrument": IC.IFEVAL_TASK, "tree_sha": tree, "dataset": IC.IFEVAL_DATASET_PATH,
        "dataset_revision": IC.IFEVAL_DATASET_REVISION, "n": IC.IFEVAL_N, "arm": C.THINKING_OFF,
        "render": {"add_generation_prompt": True, C.CHAT_TEMPLATE_KWARGS_KEY: B.render_kwargs(C.THINKING_OFF)},
        "messages_source": {CLASS_BARE: IFM.MESSAGES_SOURCE, CLASS_PROMPTED: IFM.PROMPTED_MESSAGES_SOURCE,
                            CLASS_CONSTITUTION: IFM.PROMPTED_MESSAGES_SOURCE},
        "system_prompt_cell": IC.CELL_PROMPTED, "system_prompt_sha16": hashlib.sha256(si.encode("utf-8")).hexdigest()[:16],
        "system_prompt_cells": dict(SYSTEM_PROMPT_CELL_OF_CLASS),
        "system_prompt_sha256_by_class": {CLASS_PROMPTED: hashlib.sha256(si.encode("utf-8")).hexdigest(),
                                          CLASS_CONSTITUTION: hashlib.sha256(si_constitution.encode("utf-8")).hexdigest()},
        "system_prompt_chars_by_class": {CLASS_PROMPTED: len(si), CLASS_CONSTITUTION: len(si_constitution)},
        "produced_by": {"script": "scripts/sdfbase/ifeval/measure_ifeval_geometry.py", "tokenizer_loader": "sisteer.models.load_tokenizer",
                        "token_count": "scripts/tinycap_oos_prompt_budget.py:prompt_token_count", "transformers": tf},
        "max_prompt_tokens": max_prompt, "max_prompt_argmax": argmax,
        "p99_prompt_tokens_pooled": p99_pooled,
        "prompt_classes": {C.IFEVAL_PROMPT_CLASS_SHORT: list(SHORT_CLASSES), C.IFEVAL_PROMPT_CLASS_LONG_SI: list(LONG_SI_CLASSES)},
        "max_prompt_tokens_longsi": max_prompt_longsi, "max_prompt_argmax_longsi": argmax_longsi,
        "p99_prompt_tokens_pooled_longsi": p99_pooled_longsi,
        "sizing_rule": f"L = next_block({sg.VLLM_BLOCK_SIZE})(max_prompt_tokens + cap) per PROMPT CLASS — "
                       f"{sg.SIZED_BY_MAX_PROMPT_PLUS_MAX} (no cap-bound draw is clamped); the short classes (bare, prompted) "
                       f"size capability:ifeval_<cap>, the constitution class sizes capability:ifeval_longsi_<cap>",
        "by_model": by_model,
        "by_slug": {slug: {"model_key": k, "view": v} for (k, v), slug in IC.IFEVAL_SLUGS.items()},
        "entries": entries,
        "entries_longsi": entries_longsi,
    }


def _seqs_at(model_key: str, L: int) -> Optional[int]:
    """The STARTING S the model's budget buys at L (the hybrid cap applied) — on Modal the KV-derived shrink decides."""
    try:
        return int(sg.budget_for(model_key).entry.seqs_at(L)) if sg.budget_for(model_key).per_model else int(sg.identity_seqs(L, budget=sg.budget_for(model_key).batched_token_budget))
    except Exception:  # noqa: BLE001 — a model with no budget row simply reports none
        return None


def render_measurement(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--tree-sha", default=None, help="the tree the measurement is recorded against (default: git HEAD)")
    ap.add_argument("--artifacts-dir", type=Path, default=None, help="also write the per-(model, class, cap) budget artifacts here (outside the repo)")
    ap.add_argument("--offline", action="store_true", help="HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1 for this call")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed measurement drifts from a regeneration at its recorded tree sha")
    a = ap.parse_args(argv)
    saved = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    try:
        if a.offline:
            for k in saved:
                os.environ[k] = "1"
        if a.check:
            prior = json.loads(Path(a.out).read_text(encoding="utf-8"))
            doc = measure(tree=prior.get("tree_sha"))
            if render_measurement(doc) != Path(a.out).read_text(encoding="utf-8"):
                print(f"DRIFT {a.out}")
                return 1
            print(f"OK {a.out}: byte-identical at tree {prior.get('tree_sha')}")
            return 0
        doc = measure(tree=a.tree_sha or tree_sha(), artifacts_dir=a.artifacts_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(render_measurement(doc), encoding="utf-8")
        for key, e in doc["by_model"].items():
            b, pr, co = e["classes"][CLASS_BARE], e["classes"][CLASS_PROMPTED], e["classes"][CLASS_CONSTITUTION]
            print(f"{key:16s} {e['chat_template_hash']}  bare max {b['max']:5d} p99 {b['p99']:5d} p50 {b['p50']:5d}   "
                  f"prompted max {pr['max']:5d} p99 {pr['p99']:5d}  SI +{e['si_delta']['min']}..{e['si_delta']['max']}   "
                  f"constitution max {co['max']:5d} p99 {co['p99']:5d}  SI +{e['constitution_si_delta']['min']}..{e['constitution_si_delta']['max']}")
        for cap, en in doc["entries"].items():
            print(f"{en['name']:32s} L = next_block({doc['max_prompt_tokens']} + {cap}) = {en['max_model_len']}  starting S {en['starting_seqs_by_model']}")
        for cap, en in doc["entries_longsi"].items():
            print(f"{en['name']:32s} L = next_block({doc['max_prompt_tokens_longsi']} + {cap}) = {en['max_model_len']}  starting S {en['starting_seqs_by_model']}")
        print(f"wrote {a.out}")
        return 0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    raise SystemExit(main())
