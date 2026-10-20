# The truncation rule (cap ladder)

Every generation pass of the capability harness is checked for truncation. A
(cell, readout) whose truncated share exceeds 1% is re-drawn at the next
larger cap, up the ladder `512 → 1024 → 2048 → 4096 → 8192 → 16384`, until the
share is at or below 1% or the 16,384 ceiling is reached; the report at exit
names every residual. Every rung is its own store at its own cap. Code:
`src/sisteer/capeval/tiers/cap_ladder.py` (the rule and its constants),
`carry_over.py` (the rung partition), the CLI `scripts/cap_ladder.py`, the
rung driver `scripts/capctrl_cap2048_plan.py`, and the generating drivers
`scripts/cell_controls_run.py` and `scripts/served_capability_run.py`.

## Definitions

| term | rule |
|---|---|
| at cap | `min(ntok(text) + 1, cap) >= cap` under the cell's own tokenizer (`carry_over.AT_CAP_RULE`) |
| degenerate (loop) | `rep_ratio > 0.5` over the whole text **or** its last 200 whitespace tokens; `rep_ratio = 1 − unique/total whitespace-token 4-grams`, under 8 words → 0 |
| boundary | an at-cap item for which another draw in hand terminated below the cap |
| truncated | at cap under the item's own source cap, not degenerate, not boundary |
| trigger | `n_truncated / n > 0.01` and `cap < 16384` |

Degenerate items are never re-drawn (a larger cap cannot fix a loop); they
are carried as `non_terminating` and reported separately. Boundary items are
out of the trigger but re-drawn when the cell climbs. Every cap the rule
reads comes from the run; only the rungs and the ceiling are its constants.

## Exits per rung

| verdict | meaning |
|---|---|
| `escalate` | share > 1% below the ceiling: climb |
| `clean` | nothing truncated |
| `below_threshold` | 0 < share ≤ 1%: the residual is listed |
| `ceiling` | share > 1% at 16,384 after climbing |
| `ceiling_at_outset` | share > 1% at a cap already at or above the ceiling (MATH-500's default): no rung ever existed |
| `no_rung` | share > 1% on a leg with no rung mechanism (the Inspect tasks): unresolved |
| `transport_refused` | the recorded hop timeouts cannot carry the next rung |
| `ledger_censored` | the producer's usage ledger holds more at-cap completions than the scored spool: unresolved |

## Commands

- `scripts/cap_ladder.py size --family F --tasks T,... --out sizing.json`
  names the smallest rung whose at-cap share is ≤ 1%; the drivers take it as
  `--cap-sizing` and refuse a smaller judged cap.
- Both drivers measure every pass, writing `private/gens_<name>.caps.jsonl`
  per item and `cap_ladder.json` per cell; `scripts/cap_ladder.py measure
  --store <cell dir>` measures an older store.
- `scripts/capctrl_cap2048_plan.py --cap R --new-cap R'` partitions each spool
  into `carried` / `regenerate` / `non_terminating` / `residual`; the served
  driver consumes the bundle with `--carry-over`.
- `scripts/cap_ladder.py report <roots...> --out report.json --md report.md`
  lists every (cell, readout)'s exit and residual.
