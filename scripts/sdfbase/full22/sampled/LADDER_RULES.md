# full22 sampled re-run — the cap-ladder rules

The token-cap ladder's decision rules as the tools implement them, the classifier with its provenance, the boundary
parameter and the record schema. Tools, under `scripts/sdfbase/full22/sampled/` unless stated:

| tool | role |
|---|---|
| `ladder_decide.py static \| controls` | the per-(cell, readout) decision + the at-cap disposition file |
| `ladder_decide.py reclassify --kind static\|controls …` | the same decision over every rung already generated, no regeneration |
| `escalate_static.py … [--items FILE]` | the static rung driver; re-draws only the disposition's escalation set |
| `scripts/served_capability_run.py generate … --controls-cap N --escalate-from-root ROOT [--items FILE]` | the controls rung driver; same semantics |
| `at_cap_static.py` / `at_cap_controls.py [--share]` | the raw at-cap count, or `{n_at_cap, n, share}` |
| `ladder_extend.py decide` | the 16,384 extend decision and the ceiling report over the reclassify records |

## 1. The ladder

- Ceiling 16,384. Controls `512 -> 2048 -> 8192 -> 16384`; static `4096 -> 8192 -> 16384` (`LADDERS` in
  `ladder_decide.py`, `CONTROLS_CAP_LADDER` in `sisteer.eval.sampled_controls`). Per item, self-limiting per (cell, readout).
- Trigger. A (cell, readout) escalates from rung N to N+1 only when `n_truncated / n_responses > 0.01`; it leaves at
  `<= 1%` (`exit_le_1pct`; exactly 1 % exits) or at the ceiling (`exit_ceiling` when the share is still above 1 % there).
- Escalate only truncated items: `truncated = at_cap AND cap_coherent AND NOT boundary`. Loops, repetitive items and
  boundary draws are copied forward, still at cap, stamped `escalation_excluded: loop | repetitive | boundary` and
  counted in `escalation.json` / `gen_stats.json`; they are reported as non-terminating with their class, never as
  unresolved truncation.
- Report at every rung, per (cell, readout): `n_responses, n_at_cap, n_loop` (by which rule fired), `n_repetitive,
  n_boundary, n_truncated, share_truncated, decision`, and for an exit with a residual its composition (§5).

## 2. The classifier — `item_filtered_v1`

Every at-cap item is classed on `rep_ratio_max = max(rep_ratio(text), rep_ratio(last 200 whitespace tokens))`; the
reference implementation is `src/sisteer/capeval/tiers/cap_ladder.py` (`TAIL_WORDS = 200`, `REP_RATIO_THRESHOLD = 0.5`,
strict `>`). Every record carries `classifier = {name, version: "item_filtered_v1", source, source_commit, source_sha,
window: "whole_or_last_200_words", tail_words, loop_threshold, loop_operator}` and `rule = "item_filtered_v1"`.

| class | rule |
|---|---|
| `loop` | `rep_ratio_max > 0.5` (strict) OR `sisteer.eval.degeneracy.classify(text)["degenerate"]`; `loop_by` ∈ rep_ratio / degeneracy / both, `loop_window` ∈ whole / tail / both |
| `repetitive` | `0.2 <= rep_ratio_max <= 0.5` and not degenerate; reported, never escalated |
| `cap_coherent` | `rep_ratio_max < 0.2` and not degenerate; the only class that may escalate |

```python
def rep_ratio(t):                      # 1 - unique / total whitespace-token 4-grams; fewer than 8 words -> 0
    w = t.split()
    if len(w) < 8:
        return 0.0
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g)
```

The function is extracted by AST at run time from the flags producer named by `--rep-ratio-source` (refused when
absent), never re-typed; `common_sampled.sh` pins that file's sha256
`152d180376dd3ce9d6e054354b8bd4a6ddcf6ffd9d4635ab708398d32c5f33ae` (function text
`4e8247137173b2d519124036b7947b430d6aa55d5727dd39ebb0226578a47940`), and both are recorded on every decision. An item at
exactly 0.5 is `repetitive` (`n_rep_ratio_at_tie` counts them). `src/sisteer/eval/degeneracy.py` is loaded by path (sha
recorded) with its constants: tail window 2000 / block 60 / ≥ 3 repeats / min length 200; line rule ≥ 8 lines, one line
≥ 25 %. Classification runs on the full stored text.

## 3. The boundary rule

An at-cap item whose sibling trial's row for the same item at the same rung terminated freely well under the cap is
`boundary` noise: `sibling.finish_reason != "length"` and `sibling.completion_tokens <= BOUNDARY_FRACTION × cap`,
`BOUNDARY_FRACTION = 0.5` (`--boundary-fraction`, recorded per decision). Boundary items are counted (`n_boundary`),
never in the trigger share; `n_at_cap = n_loop + n_repetitive + n_boundary + n_truncated` (a loop that is also boundary
counts as a loop; `n_boundary_any` applies the sibling rule to every at-cap item). The sibling is named explicitly
(`--sibling-private-run` / `--sibling-out-root`); when it is absent, incomplete, a different item set or at a different
cap, no item is called boundary, the decision is still valid (exit 0) and the record says `sibling.status =
"unavailable"` with the reason (`not_given` when unnamed). A sibling row copied forward from a lower rung terminated at
that cap, which satisfies the rule.

## 4. Finer loop classes

The disposition file keeps per-item `rep_ratio`, `tail_rep_ratio`, `rep_ratio_max` and `degenerate_reason`, so a finer
loop taxonomy can be applied afterwards without re-deciding anything; it could not change the escalation set.

## 5. The record schema

`decision.json`, one per (cell, readout, rung): `tool` {name, path, sha256, git_sha, argv, written_at}; `rule`;
`classifier` (§2); `kind` static | controls; `identity` (static: private_root, private_run, attack, defense, trial,
store, n_shards, is_rung; controls: out_root, store, cell, control); `cap`, `ladder`, `ceiling`, `next_cap` (null unless
escalating), `at_ceiling`; `rules` {trigger_share 0.01, loop_rep_ratio_threshold 0.5, repetitive_rep_ratio_min 0.2,
tail_words 200, boundary_fraction 0.5, the rule texts}; `rep_ratio` {source, source_sha256, rep_ratio_source,
rep_ratio_source_sha256}; `degeneracy` {source, source_sha256, rules}; `sibling` {status used | unavailable |
not_given, reason, ...}; `counts` {n_responses, n_at_cap, n_loop, n_loop_by_rule {rep_ratio, degeneracy, both},
n_loop_by_window {whole, tail, both}, n_rep_ratio_at_tie, n_repetitive, n_boundary, n_boundary_any,
n_sibling_unreadable, n_truncated, share_truncated, share_at_cap}; `decision` escalate | exit_le_1pct | exit_ceiling;
`n_truncated` and `share_truncated` again at top level for the shell caller; `residual` (null, or the at-cap
composition left at an exit); `items_out`, `items_out_sha256`, `n_items_escalate`. stdout carries the same record as
one JSON line minus `rep_ratio.rep_ratio_source`. Exit 0 on any valid decision; exit 2 on a refusal (a greedy-shaped,
torn, incomplete or missing store, a cap that is not `--cap` or off the ladder, an absent rep_ratio source).

The at-cap disposition file (`--items-out`; the drivers' `--items`): a header `{"__ladder_items__": {schema: 2, kind,
cap, ...identity..., n_at_cap, n_escalate, decision, rule, classifier, tool_sha256, rep_ratio_source_sha256}}`, then
one row per at-cap item: `goal_id, sample_idx, trial` (static) or `control, item_id` (controls), plus `class,
rep_ratio, tail_rep_ratio, rep_ratio_max, loop_by, loop_window, degenerate_reason, boundary, sibling, escalate,
excluded_as`; `escalate` is true iff `excluded_as` is null. A driver given `--items` requires a row for every at-cap
source item, refuses rows naming items not at cap, checks the header's cap and identity, refuses a header whose
`rule` is not `item_filtered_v1`, re-draws exactly the `escalate: true` rows and copies the rest forward with
`escalation_excluded = excluded_as`. Without `--items` both drivers re-draw every at-cap item.

`reclassify` walks every rung present under `--root` (static: `<run>`, `<run>_cap8192`, `<run>_cap16384`; controls:
`--root-pattern "controls_cap{cap}/{tag}_t{trial}"`), writes `DIR/static/<run>/<attack>/<defense>/{decision.json,
items.jsonl}` / `DIR/controls/cap<N>/<cell>/<control>/{…}` and `DIR/reclassify.json` (`rungs_present`, `rungs_absent`,
`n_decided`, `n_refused`, `n_by_decision`, `entries`, `complete`). An absent rung is listed, not an error; a store that
refuses is listed with its reason and the pass exits 2. The extend pass reads `decision` / `next_cap` per entry and
asserts `rule == "item_filtered_v1"`.

## 6. The capability readouts

MMLU runs unchanged under a sampled config: a teacher-forced letter-logprob argmax that no sampler setting can move;
the store says so (`gen_stats["mmlu"].sampled_decode_applies = false`), a rung refuses it and `at_cap_controls` never
reads it as a sampled control. A root under `sampled/mmlu/` is therefore a canary, not a result: its per-item letter
agreement across trials measures server-to-server logprob nondeterminism (`analysis_sampled.py mmlu-agreement`, which
marks its output `canary_record` and `is_paper_result: false`), and it is never presented as the paper's MMLU number.
The sampled thinking-ON GSM8K readout that `sampled_decode.readouts.gsm8k` describes is retired and no job runs it; the
static families and judged controls ignore the `readouts` sub-block.
