#!/usr/bin/env python
"""seedrep_ladder_helpers.py — the base container's ladder TAILER runs these under the DRIVER venv (the app's own python has
no `sisteer`): the exact request the misalign runner sent for an at-cap row, and a prompted arm's SI text — the two facts
a queue record's `redraw` block needs that no durable stream carries.

  misalign-messages --family agentic --cells <cells.json> --requests <json [{cell_id, behavior, item_id}]> --out <json>
      -> {"<cell_id>|<behavior>|<item_id>": messages} through the runner's OWN `load_items` (a `(items, suite_revision)` TUPLE, unpacked
         the way the runner does) / `si_text_for` / `build_messages` (the composition `_run_cell` uses), so the rung
         re-issues byte-identical messages; the suite revision is logged to stderr (the queue record carries no slot for it).
  si-text --si-id <harness si id>          -> the SI text on stdout (`misalign.runner.si_text_for`, the harness SI vocabulary)
  controls-prompts --control <xstest|or_bench_hard_1k> --items <json list of item ids> --out <json>
      -> {item_id: prompt} through `sisteer.capeval.tiers.judged.xstest_rows` / `or_bench_rows` (the pinned corpora rows the served
         controls driver itself draws; item ids are content ids, so the lookup is exact); an id no row carries is REFUSED (rc 3).
  expected-samplers --row <plan row json> --tree <W checkout> --out <json>
      -> the row's CLIENT tasks with the sampler each is EXPECTED to send (`record_resolved_sampler.py --expect-tasks`), one
         entry per task: `caps` (the max_tokens its requests carry), `sampler` (the frozen keys) and `source`. Controls: the sampled
         over-refusal tasks from the child's config through `controls_gen_config` (as `controls-sampling`), MMLU greedy at max_tokens 1
         (the driver's letter-read payload); BoN and every other leg their battery decode ref (`seedrep_launch_plan.BATTERY_DECODE_REFS`,
         the rows the certificate's decode proof binds) resolved by EXACT ref through the tree's registry. A task
         whose reference cannot be resolved is written with `sampler: null` and `why` -- the recorder then reads it UNCOMPARED BY NAME,
         never as agreeing -- so the helper exits 0 whenever it could read the row (rc 3 only when the row itself is unreadable).
  controls-sampling --config <config rel path> --out <json>
      -> `sisteer.eval.cell_controls.controls_gen_config(cfg)` -- the controls decode row the driver keyed its stores on (the sampled
         block's policy row), never re-implemented here -- PROJECTED to the frozen sampler-key set (the sibling protocol module's
         `seedrep_ladder.sampler_keys()`, the tuple the queue validator reads): exactly the six vendor terms a queue record's
         `redraw.sampling` may carry. The driver's row
         also carries the store's bookkeeping (`arm, do_sample, max_new_tokens, seed_base, trial`); written unprojected,
         `Producer.enqueue` -> `check_redraw_sampling` refused the first at-cap record as "extra" keys and the base row failed
         LADDER_TAILER_ERROR. A frozen key the resolver did not return is REFUSED here by name (rc 3), never defaulted server-side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def misalign_messages(family: str, cells_path: Path, requests: list, out: Path) -> int:
    from sisteer.capeval.misalign import runner as mr

    cells, _extra = mr.load_cells_file(cells_path)
    by_id = {c["cell_id"]: c for c in cells}
    behaviors = sorted({str(r["behavior"]) for r in requests if r.get("behavior")}) or None
    # `load_items` returns `(items, suite_revision)` -- the runner unpacks it (`items, revision = load_items(...)`); iterating the TUPLE
    # instead made every store's tailer refuse `helper rc 1 ... 'list' object has no attribute 'get'` after 5 children rc 0.
    # The queue record has no revision slot (identity = the row's keys; redraw = vmodel, messages, sampling), so it is logged, not returned.
    loaded_items, suite_revision = mr.load_items(family, behaviors)
    items = {(str(it.get("behavior")), str(it["item_id"])): it for it in loaded_items}
    print(f"misalign-messages: {family} suite {suite_revision}: {len(items)} item(s) for behaviours {behaviors or 'all'}; "
          f"{len(requests)} request(s)", file=sys.stderr)
    si_cache: dict = {}
    result = {}
    missing = []
    for r in requests:
        cell = by_id.get(r["cell_id"])
        key = f"{r['cell_id']}|{r.get('behavior')}|{r['item_id']}"
        item = items.get((str(r.get("behavior")), str(r["item_id"])))
        if cell is None or item is None:
            missing.append(key)
            continue
        si_id = cell.get("si_id")
        if si_id and si_id not in si_cache:
            si_cache[si_id] = mr.si_text_for(si_id)
        result[key] = mr.build_messages(family, item, si_cache.get(si_id) if si_id else None)
    out.write_text(json.dumps(result, ensure_ascii=False))
    if missing:
        print(f"misalign-messages: {len(missing)} request(s) unresolved: {missing[:5]}", file=sys.stderr)
        return 2
    return 0


def controls_prompts(control: str, item_ids, out: Path, *, rows_of=None) -> int:
    """{item_id: prompt} for the ids asked, from the judged corpora's pinned rows (`rows_of(control)` defaults to the tree's loaders)."""
    if rows_of is None:
        from sisteer.capeval.tiers import judged
        rows_of = {"xstest": lambda: judged.xstest_rows(), "or_bench_hard_1k": lambda: judged.or_bench_rows()}.__getitem__
    by_id = {str(r["item_id"]): str(r["prompt"]) for r in rows_of(control)()}
    wanted = [str(i) for i in item_ids]
    missing = [i for i in wanted if i not in by_id]
    if missing:
        sys.stderr.write(f"controls-prompts: {len(missing)} item id(s) no pinned {control} row carries: {missing[:5]}\n")
        return 3
    out.write_text(json.dumps({i: by_id[i] for i in wanted}, indent=1))
    return 0


def _queue_sampler_keys() -> tuple:
    """The frozen sampler-key tuple the queue VALIDATOR reads (`seedrep_ladder.sampler_keys()`), from the sibling protocol module loaded
    by path (the helper runs from the staged tree, beside it; a unique module name so a test's own copy is never shadowed)."""
    import importlib.util as ilu
    path = Path(__file__).resolve().with_name("seedrep_ladder.py")
    spec = ilu.spec_from_file_location("seedrep_ladder_for_helpers", path)
    mod = sys.modules.get(spec.name)
    if mod is None:
        mod = ilu.module_from_spec(spec)
        sys.modules[spec.name] = mod              # registered BEFORE exec: the module's dataclasses resolve their namespace through sys.modules
        spec.loader.exec_module(mod)
    return tuple(mod.sampler_keys())


def _controls_sampling_row(config: Path, *, gen_of=None) -> dict:
    """The config's controls decode row (`controls_gen_config`) projected to the frozen sampler-key set; raises ValueError naming a missing key."""
    import yaml
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if gen_of is None:
        from sisteer.eval.cell_controls import controls_gen_config
        gen_of = controls_gen_config
    keys = _queue_sampler_keys()
    gen = gen_of(cfg)
    if not isinstance(gen, dict) or not gen:
        raise ValueError(f"{config}: controls_gen_config returned no row")
    sampling = {k: gen[k] for k in keys if k in gen}
    missing = [k for k in keys if k not in sampling]
    if missing:
        raise ValueError(f"{config}: the controls decode row lacks frozen sampler key(s) {missing} (row keys {sorted(gen)})")
    return sampling


#: the MMLU letter readout's request (`scripts/served_capability_run.py::_mmlu_scoring_payload`: `temperature 0.0, max_tokens 1, seed 0`; every
#: other sampler key omitted -> the server's defaults, which the reference therefore does not pin) -- pinned equal by test
MMLU_EXPECTED_SAMPLER = {"temperature": 0.0}
MMLU_SCORING_CAP = 1
EXPECT_TASKS_SCHEMA = "resolved_sampler_task_expectations/1"       # record_resolved_sampler.EXPECT_TASKS_SCHEMA, pinned equal by test


def _plan_module():
    """`seedrep_launch_plan` (stdlib-only) loaded by path beside this helper, under a unique name (the same pattern as `_queue_sampler_keys`)."""
    import importlib.util as ilu
    path = Path(__file__).resolve().with_name("seedrep_launch_plan.py")
    spec = ilu.spec_from_file_location("seedrep_launch_plan_for_helpers", path)
    mod = sys.modules.get(spec.name)
    if mod is None:
        mod = ilu.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return mod


def _decode_ref_class(eval_leg: str, lp) -> "str | None":
    """The `BATTERY_DECODE_REFS` class a plan child's eval leg draws under (None for the legs with their own resolver: controls, bon)."""
    if eval_leg in lp.STATIC_LEGS:
        return "static"
    if eval_leg in lp.MULTITURN_LEGS:
        return "multi_turn"
    if eval_leg == "petri":
        return "petri"
    if eval_leg.startswith("misalign_"):
        return "agentic"
    if eval_leg == "gsm8k":
        return "gsm8k"
    return None


def _registry_sampler(tree: Path, ref: str) -> dict:
    """The registry row `ref` (EXACT ref) through the tree's export, projected to the frozen sampler keys (only the keys the row SENDS)."""
    from sisteer.registry.decode_policy import PolicyFile
    pf = PolicyFile.load(str(Path(tree) / "configs/registry/decode_policy.json"))
    pol = pf.by_ref(ref)
    if getattr(pol, "superseded_by_policy_id", None) is not None:
        raise ValueError(f"registry row {ref!r} is superseded by {pol.superseded_by_policy_id!r}")
    params = pol.request_params()
    keys = _queue_sampler_keys()
    return {k: params[k] for k in keys if k in params and params[k] is not None}


def gsm8k_probe_block(kids, lp, *, registry_sampler) -> "dict | None":
    """The gsm8k children's RENDER PROBE -- `served_capability_run._render_probe`, ONE 1-item request per thinking arm at
    RENDER_PROBE_MAX_TOKENS 512 decoding under the cell's frozen thinking-card row (F-AJ's render certification reads the same card) -- lands
    in the recorder's PROBE class (cap 512 is a pinned probe cap) and read UNCLASSIFIED_REQUESTS under the default greedy probe reference (the
    gsm8k evidence: `client UNCOMPARED n=1 ... max_tokens=512`). The recorder's `kinds` mode classifies it BY THE DRIVER'S OWN TAG
    (`X-Tinycap-Task: probe/render`, written to each child's `<out-root>/request_evidence.jsonl` -- the location contract,
    `seedrep_launch_plan.GSM8K_OUT_REL`) and compares it against the pinned card row (`--expect`); the greedy bites at 48 are tagged
    `probe/bites` by the same writer. Everything here is DERIVED from the plan row: the card row is the registry's gsm8k thinking-card ref
    (the row's `thinking` card, `BATTERY_DECODE_REFS["gsm8k"]`), the tag files the children's declared store roots. None for a row without a
    gsm8k child (the controls / static / bon paths are byte-identical)."""
    g = [k for k in kids if str(k.get("eval_leg") or "") == "gsm8k"]
    if not g:
        return None
    ref = lp.BATTERY_DECODE_REFS["gsm8k"]
    try:
        expect = registry_sampler(ref)
    except Exception as e:  # noqa: BLE001 -- recorded, never invented: without a card reference the recorder REFUSES kinds mode (exit 2), loudly
        return {"probe_expect": "kinds", "expect": None, "why": f"{type(e).__name__}: {e}", "tags_rel": [], "source": f"registry {ref}"}
    tags_rel = []
    for k in g:
        cap, s_, cell = k.get("cap"), k.get("s"), k.get("cell") or k.get("arm")
        if cap is None or s_ is None or not cell:
            continue
        tags_rel.append(f"{lp.GSM8K_OUT_REL.format(cap=int(cap), s=int(s_), cell=cell)}/request_evidence.jsonl")
    return {"probe_expect": "kinds", "expect": expect, "tags_rel": sorted(set(tags_rel)),
            "source": f"render: registry {ref} (exact ref; the row's thinking card); tags: the gsm8k children's <out-root>/request_evidence.jsonl (GSM8K_OUT_REL)"}


def expected_samplers(row: dict, tree: Path, out: Path, *, controls_row_of=None, registry_sampler=None) -> int:
    """The per-task expectation document for ONE plan row (`record_resolved_sampler.py --expect-tasks`): every client task the row's children
    send, with the caps their requests carry and the sampler each is expected to send -- resolved from the SAME authorities the drivers use,
    never typed. A task whose reference cannot be resolved (an unknown leg, a config / registry refusal, children disagreeing) is written with
    `sampler: null` and `why`: the recorder reads it UNCOMPARED by name."""
    lp = _plan_module()
    controls_row_of = controls_row_of or (lambda cfg: _controls_sampling_row(Path(tree) / cfg))
    registry_sampler = registry_sampler or (lambda ref: _registry_sampler(tree, ref))
    kids = row.get("children")
    if not isinstance(kids, list) or not kids:
        sys.stderr.write(f"expected-samplers: row {row.get('standup_id')!r} carries no children list\n")
        return 3
    found: dict = {}          # task -> list of (caps, sampler | None, source, why)

    def add(task, caps, sampler, source, why=None):
        found.setdefault(str(task), []).append(([int(c) for c in caps], sampler, source, why))

    for k in kids:
        leg = str(k.get("eval_leg") or "")
        cap = k.get("cap")
        try:
            if leg == "controls":
                tasks = k.get("task") if isinstance(k.get("task"), dict) else {}
                if not tasks:
                    raise ValueError("controls child carries no {control: task} map")
                for control, task in tasks.items():
                    if control in lp.CONTROLS_LADDER:
                        cfg = k.get("config")
                        if not cfg or cap is None:
                            raise ValueError(f"controls child ({control}) carries no config / cap")
                        add(task, [cap], controls_row_of(cfg), f"controls_gen_config({cfg}) projected to seedrep_ladder.sampler_keys()")
                    elif control in lp.CONTROLS_FIXED_CAP:
                        add(task, [MMLU_SCORING_CAP], dict(MMLU_EXPECTED_SAMPLER),
                            "served_capability_run._mmlu_scoring_payload: temperature 0.0 at max_tokens 1 (teacher-forced letter read; other keys omitted, not compared)")
                    else:
                        add(task, [cap] if cap is not None else [], None, "", f"unknown control {control!r}")
            elif leg == "bon":
                # the registry's vendor row by EXACT ref (`BATTERY_DECODE_REFS["bon"]`, the row the certificate's decode proof binds) -- what the
                # driver's `bon_served.bon_policy(model, "vendor_preset")` resolves to; never that module (scripts/modal stays off bon_served)
                if cap is None:
                    raise ValueError("bon child carries no cap")
                ref = lp.BATTERY_DECODE_REFS["bon"]
                add(k.get("task") or "strongreject_bon", [cap], registry_sampler(ref), f"registry {ref} (exact ref) request_params projected to seedrep_ladder.sampler_keys()")
            else:
                cls = _decode_ref_class(leg, lp)
                task = k.get("task") or leg
                if cls is None:
                    add(task, [cap] if cap is not None else [], None, "", f"no expected-sampler resolver for eval leg {leg!r}")
                    continue
                if cap is None:
                    raise ValueError(f"{leg} child carries no cap")
                ref = lp.BATTERY_DECODE_REFS[cls]
                add(task, [cap], registry_sampler(ref), f"registry {ref} (exact ref) request_params projected to seedrep_ladder.sampler_keys()")
        except Exception as e:  # noqa: BLE001 -- recorded on the task, never dropped: the recorder reads it UNCOMPARED by name
            task = k.get("task") if not isinstance(k.get("task"), dict) else ",".join(str(t) for t in k["task"].values())
            add(task or leg or "?", [cap] if isinstance(cap, int) else [], None, "", f"{type(e).__name__}: {e}")
    tasks: dict = {}
    for task, specs in found.items():
        caps = sorted({c for cs, _s, _src, _w in specs for c in cs})
        samplers = {json.dumps(s_, sort_keys=True) for _c, s_, _src, _w in specs if s_ is not None}
        whys = [w for _c, _s, _src, w in specs if w]
        sources = sorted({src for _c, _s, src, _w in specs if src})
        if whys or not samplers:
            tasks[task] = {"caps": caps, "sampler": None, "source": "; ".join(sources), "why": "; ".join(whys) or "no reference resolved"}
        elif len(samplers) > 1:
            tasks[task] = {"caps": caps, "sampler": None, "source": "; ".join(sources), "why": f"the row's children disagree on the sampler ({sorted(samplers)})"}
        else:
            tasks[task] = {"caps": caps, "sampler": json.loads(next(iter(samplers))), "source": "; ".join(sources)}
    doc = {"schema": EXPECT_TASKS_SCHEMA, "standup_id": row.get("standup_id"), "leg": row.get("leg"), "stage": row.get("stage"), "tasks": tasks}
    probe = gsm8k_probe_block(kids, lp, registry_sampler=registry_sampler)
    if probe is not None:
        doc["probe"] = probe
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, default=str))
    unresolved = sorted(t for t, spec in tasks.items() if spec["sampler"] is None)
    print(f"expected-samplers: {row.get('standup_id')}: {len(tasks)} task(s) -> {out}" + (f"; UNRESOLVED (the recorder reads them UNCOMPARED): {unresolved}" if unresolved else ""),
          file=sys.stderr)
    return 0


def controls_sampling(config: Path, out: Path, *, gen_of=None) -> int:
    """The config's controls decode row through the driver's OWN resolver (`controls_gen_config`), PROJECTED to the frozen sampler-key
    set -- `seedrep_ladder.sampler_keys()`, the ONE tuple `Producer.enqueue` -> `check_redraw_sampling` validates a queue record's
    `redraw.sampling` against (in its order), so the helper's output IS the block the record carries. The resolver's bookkeeping
    keys (`arm, do_sample, max_new_tokens, seed_base, trial`) are the STORE's identity, not the wire's, and the strict validator
    refuses them as "extra" at enqueue. A frozen key the row lacks refuses by name (rc 3): that term would run
    under the server's defaults, invisible to the assembly."""
    try:
        sampling = _controls_sampling_row(config, gen_of=gen_of)
    except ValueError as e:
        sys.stderr.write(f"controls-sampling: {e}; a re-draw without them would run under the server's defaults\n")
        return 3
    out.write_text(json.dumps(sampling, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("misalign-messages")
    m.add_argument("--family", default="agentic")
    m.add_argument("--cells", type=Path, required=True)
    m.add_argument("--requests", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("expected-samplers")
    e.add_argument("--row", type=Path, required=True, help="the plan row (JSON: leg, standup_id, children[] with eval_leg / task / cap / config)")
    e.add_argument("--tree", type=Path, required=True, help="the W checkout (configs + configs/registry/decode_policy.json)")
    e.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("si-text")
    t.add_argument("--si-id", required=True)
    cp = sub.add_parser("controls-prompts")
    cp.add_argument("--control", required=True, choices=("xstest", "or_bench_hard_1k"))
    cp.add_argument("--items", type=Path, required=True)
    cp.add_argument("--out", type=Path, required=True)
    cs = sub.add_parser("controls-sampling")
    cs.add_argument("--config", type=Path, required=True)
    cs.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "misalign-messages":
        return misalign_messages(a.family, a.cells, json.loads(a.requests.read_text()), a.out)
    if a.cmd == "controls-prompts":
        return controls_prompts(a.control, json.loads(a.items.read_text()), a.out)
    if a.cmd == "controls-sampling":
        return controls_sampling(a.config, a.out)
    if a.cmd == "expected-samplers":
        try:
            row = json.loads(Path(a.row).read_text())
        except (OSError, ValueError) as e:
            sys.stderr.write(f"expected-samplers: --row {a.row}: unreadable ({type(e).__name__}: {e})\n")
            return 3
        if not isinstance(row, dict):
            sys.stderr.write(f"expected-samplers: --row {a.row}: not a plan row object\n")
            return 3
        return expected_samplers(row, a.tree, a.out)
    from sisteer.capeval.misalign import runner as mr
    sys.stdout.write(mr.si_text_for(a.si_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
