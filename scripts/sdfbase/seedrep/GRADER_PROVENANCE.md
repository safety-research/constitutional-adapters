# GRADER_PROVENANCE — the seed-replicate copy of the ReLU-native grader

`scripts/sdfbase/seedrep/grade_full22_native.py` is a copy of `scripts/sdfbase/full22/relu_native/grade_full22_native.py`
at commit `790ef2d8d428db9698c4727bfe152e13320d8369` (source sha256
`7141dd67a1f165e923dcf4670b6d0956bc1ab902826d8a00d4a0d2da80ea9a8f`, 759 lines), the commit the native wave was graded
from. The source is not on this tree and the lens graders refuse native stores by design, so the seed-replicate views'
native stores are graded by this twin. `GRADER_DIFF.patch` (`diff -u` source → copy; +1950 / −61 lines in 24 hunks,
counted as `grep -c '^+[^+]'` / `'^-[^-]'` / `'^@@'`) is the whole difference: reverse-applied to the twin it yields the
source at the sha256 above, forward-applied to that source it yields the twin (`tests/test_seedrep_grader_twin.py`). It
is regenerated from the recorded source blob whenever the twin changes.

## What the copy adds

Every hunk is additive except the BoN judge's construction, guard and grade-call lines (items 4–6), the credential
plumbing the vault mode retired (item 11) and `--org`'s help text, which names a sisteer.vault pool.

1. A copy header, `import re`, and the worktree-depth default `parents[4]` → `parents[3]` (evaluated eagerly by the
   source, so the shallower copy would raise at import).
2. `TARGETS` gains the roster's `qwen35_4b` seed-replicate views, enumerated from `configs/sdfbase_full22_roster.yaml`
   (`seedrep_view_targets()`: every view matching `^s[0-9]+$` → `qwen35_4b_s<N>`, run `sdfbase_full22_<TAG>`, default
   attacks `bon` + the three multi-turn legs), and the per-seed Arditi wave configs
   `configs/seedrep/<base>_seedrep_s<N>.yaml` (`seedrep_arditi_targets()`; a colliding or shared stem lands in
   `ARDITI_TARGETS_PROBLEMS`, not in `TARGETS`). Never a hardcoded list.
3. `--seed-ledger <seed_ledger.jsonl>` (seed-replicate targets only): each aggregate row carries `seed_rule` /
   `seed_base` = the ledger's attested values for (view, eval_id, cell), else null with the reason in
   `seed_ledger.status`; the declared rule rides beside as `seed_ledger.declared`. Read once before any spend.
4. `--rubric v1|v2`, the StrongREJECT rubric of the BoN leg: no BoN leg → V1 as the source; BoN alone → V2 by default
   (cache kind `sr2`), V1 by explicit flag; BoN beside multi-turn legs at V2 refuses. A grades root is labelled by
   `BON_RUBRIC.json`, written before any spend; a run under the other rubric refuses by name.
5. A whole-run preflight refuses when the other rubric's cached verdicts exist for this run's BoN items.
6. `BoundedJudgeClient`: at most two attempts per BoN judgment (SDK `max_retries=0` + one retry; a 429 throttles the
   gate), then `JudgeAttemptsExhausted` → UNSCORED, which fails that CELL's grade loudly (`BonCellUnscored`, a
   `SystemExit`: no partial aggregate, nothing filed for the cell). On the from-disk path (one call over every
   cell) that is the run's terminal line, as before. Under the registry worklist the per-cell loop HOLDS a cell's
   refusal (its judgment closes `partial`), grades and files the other claimed cells in the same dispatch, and raises
   the refusal once after the loop naming every unscored cell (`bon_unscored_refusal`), keeping the live grader's
   "still UNSCORED after live grading (bounded retries)" phrase on the one terminal line the grade daemon classifies
   as retryable; the next dispatch re-issues only the unscored items (from the cache) and files those cells. Before
   this the first unscored cell ended the whole dispatch, so a battery of N cells took about N dispatches. The dispatch
   that completes a battery REBUILDS the aggregate rows of every configured BoN cell an earlier dispatch filed
   (`bon_cells_judged_complete`, read before the gates) from the judge cache — a cache-only read through the grader's own
   `grade()`, with the judge client's counts pinned unchanged; a rebuilt cell not wholly in the cache refuses by name
   before any judge call — so the BoN aggregate covers every configured cell whose judgment is complete, not only the
   cells claimed in that dispatch (the aggregate names the rebuilt cells with their judgments under
   `bon_cells_rebuilt_from_cache`, the judged ones under `bon_cells_judged_this_dispatch`, and the configured cells it
   leaves out with the registry's reason under `bon_cells_not_rebuilt`; registry-worklist BoN aggregates only). The
   candidates are the config's explicit_cells minus the cells claimed in the dispatch — never the `--cells` subset, so a
   battery graded in subsets completes on its last one — and a candidate is rebuilt ONLY as the exact complement of the
   registry's own pending rule for this dispatch's judge, its objects asked again rather than restated: no pending run a
   dispatch over the cell's store would grade (the worklist's own `pending` over the whole registry for `--judge` or each
   run's instrument judge, located to the cell as `registry_worklist_pairs` locates a row, decided per store by the same
   currency rule, `store_runs_head` — a non-current covered twin decides nothing, a refusal rebuilds nothing), and a run
   of the store that is current, not superseded, in a judgeable state, not listed by `v_pending_judgment` for that judge
   (`pending.not_pending_why`) and carrying a complete, identity-current judgment BY THAT JUDGE that covers it
   (`schema_def.judgment_covers_sql` + `judgment_identity_current_sql`). A `coverage_short` judgment, a complete judgment
   by another judge only, an `identity_stale` one or a retired run never rebuilds a cell; each is named once with why,
   and `post` reads the TAG pending over it. Before this the retry of a held cell wrote an
   aggregate holding the held cell alone — and the pre-hold code the same whenever the first unscored cell was not the
   first claimed cell — so `grade_battery.sh post`, which publishes a TAG only when the aggregate holds exactly the
   config's explicit_cells, left the battery pending with no pending run left to repair it. A judge
   response cut at `max_tokens` is its own named terminal failure of that one attempt in
   `grade_bon_live._forced_tool_grade` (`stop_reason=max_tokens`, told to an optional `on_failure` sink, which every
   terminal failure of a judgment names once — a schema failure as `schema_validation:<ExcType>`, the vocabulary at
   `FAILURE_SINK_ATTR`), never a parsed value and never counted as a schema-validation failure.
7. `grading_provenance.json` records the ledger, the copy (`lane_copy`: source path, commit, sha256), the BoN rubric and
   the retry policy.
8. `PHASE_VIEW_RE` is byte-equal to the other copies pinned by `tests/test_s2r_phase_view_grammar.py` (`s<N>`,
   `s<N>r`, `s<N>rl`, `s<N>p`, `d<dose>[_relu[l]]`), so every roster view resolves by name — and the roster / models-table
   enumerations and the unknown-target listing classify a view through the models table's ONE reader (`_models_reader` /
   `_is_phase_view`: a grammar name for EVERY key — the paper's roster views stay phase views whether or not the table
   carries the key — plus a row's DECLARED WAVE, `reproduce/lib/models.phase_views_of`: the worked example's `battery`), so
   `--target qwen35_4b_demo_battery` resolves and every roster target the grammar admitted before still does.
9. Registry worklist currency: `registry_worklist_pairs` reads `v_run_current` for every located pending run. A store
   with exactly one current run is graded once under it and its other pending runs are covered, not claimed
   (`grading_provenance.json` `registry.covered_runs`); a non-current sole pending run is covered by the run its store
   was graded under; more than one current refuses by name, and so does an AMBIGUOUS rung -- a rung (token_cap_tokens,
   pass_label) registered more than once with no current run of that rung among them and no complete judgment of it on the
   store, whatever the store's other rungs do -- the rung and its ids named; every other rung is read one by one: a rung
   of a declared ladder under its rung rule alone is graded on its own reading, the rest are retired runs with the reason
   named, and a run whose rung's covering judgment is another run's and still to fold is skipped with that reason named
   (g37-rung, g37-rung-floor, g37-rung-later, g37-rung-guard, g37-rung-guard-r5). `pending_beside_selection` brings every
   pending registration of a selected store into that check.
10. The smoke goal-shard knob (the worked example's smoke): under the registry worklist a multi-turn pair whose run row is
    stamped `extra.run_overrides.keys.OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N` (the model's overrides document the planner read,
    stamped at registration) is graded in GOALSET mode over the config's eval set dealt `goals[s::N]` across shards lo..hi
    (`sisteer.eval.smoke_goals`, the planner's own dealing rule), with the store's shards held to exactly lo..hi and the
    registration's producer key (`sh<shards>of<N>`) required to spell the same shards; `--goalset-dir` beside a stamped
    run refuses (one source). The BoN leg never takes the knob. An unstamped run, and the from-disk path, grade exactly as
    before; the aggregate rows and `grading_provenance.json` carry the stamp's provenance under `goalset`.
11. The credential is `sisteer.vault`'s, as a MODE (`SISTEER_SECRETS_MODE`), after the org's move to workload identity
    federation left the API keys rejected: the judge's client is built by
    `LiveStrongREJECTGrader(org=)` through `vault.anthropic_client(org)` — the org's own key under `op` / `env`, a
    federated KEYLESS client under `wif` — and the source's secret-manager-CLI shim for a pre-442d1947 `grade_bon_live`
    (`_op_secret` / `_load_key_patched` / `OLD_KEY_PLUMBING`, which re-bound `grade_bon_live.load_key`) is gone, the one
    non-additive hunk outside the BoN judge. `grading_provenance.json.judge` carries the pool stamp the grader resolved
    (`api_pool`: `{pool, key_sha256_12}` under op / env, `{pool, auth: wif, ids_sha12}` under wif) and `auth`.
12. The process-environment pin at import is the vault's ONE rule (`vault.pin_process_environment()`): under
    `op` / `env` byte for byte the source's block (`ANTHROPIC_BASE_URL` exported as the API host, the shell's key and
    bearer popped); under `wif` nothing is set or exported and the inherited `ANTHROPIC_*` names are dropped by name. And
    `__main__` runs `main()` under `vault.credential_exit`: a credential rejection ends the process with the typed line
    the grading daemon classifies, never a traceback. The second non-additive hunk outside the BoN judge (the source's
    three `os.environ` lines become the one vault call).
13. `BoundedJudgeClient` checks the grader's credential latch before EVERY attempt, its one retry's wake included:
    the live grader's latch rides the raw client the bounded client wraps (`vault.client_credential_latch`), so
    a sibling worker's 401 / 403 filed while this worker slept in its backoff stops the retry instead of letting it issue
    one more request on the rejected credential. Inside item 6's hunk; the two-attempt contract without a held latch is
    unchanged.
14. The BoN reading by RUNG: under the registry worklist every BoN pair's rung -- its run row's `token_cap_tokens`,
    `pass_label`, `token_cap_ladder_json` -- is read BEFORE the plan and before any claim (`grade_bon_live.bon_rungs_of`), and
    selects the reading. A cap-512 rung (base_cap, or a single-rung / unladdered cap-512 row) reads the store's cap-512
    draws and files the paper's 512 set: the headline under the EVAL's own `headline_metric_id` (`asr`, checked against the
    eval row again before `scope.metrics`), the `asr_at_n{N}` curve, `bon_nauc`, the four degeneracy shares over every draw
    of the store by the tree's classifier (`bon_climb.classify_draw`), and the measured at-cap / loop `run_degeneration`
    rows (`scope.degeneration`); a native 4,096 pass files the rows the emitter always filed, and is admitted only over a
    store generated at 4,096: the config's `bon.max_new_tokens`, and every draw's `max_tokens_clamped_from` where one is
    recorded, must equal the rung's cap, else the run is refused by name before any claim, naming the run and both caps
    (`grade_bon_live.bon_native_cap_refusal`; a cap-4,096 registration with no ladder over a 512-token store is otherwise
    read as a native pass); a CLIMB (cap 4,096 / ladder_rung) is graded on the climb fold of item 15 and is REFUSED by name
    with nothing claimed when its shards are not located or do not verify -- dry (the plan marks the pair REFUSED, then
    refuses) and real; a cap-512 rung whose eval declares `asr_bon_cap4096` refuses the same way. `grading_provenance.json`
    records every pair's rung and reading (`registry.bon_rungs`), and the g37-rung entries name the reading beside the rung.
15. The BoN CLIMB FOLD: a climb -- recognised from its OWN row (cap 4,096 / ladder_rung, or a ladder naming 512) OR, for a
    legacy row (cap 4,096 / no pass label / ladder []), from its INSTRUMENT (`read_at ladder_final` over a ladder naming 512
    at cap 4,096; never a native pass) -- is graded on its OWN reading: the 512 store's draws with each climbed at-cap draw
    REPLACED by its 4,096 redraw, loops and every other draw kept at 512 (analyze_bon_cap4096's counterfactual rule), the
    folded draws judged through the SAME judge path and cache (only the redrawn texts are new keys). Its shards are LOCATED
    by record (`run.extra_json.climb_root`) or by an explicit per-run operator override (`--bon-climb-root <run_id>=<root>`,
    repeatable) and VERIFIED before any claim (`grade_bon_live.bon_climb_inputs`: the plan's `store_dir` is the run's
    store; EVERY manifest row's `response_512_sha256`, loops included, is the store's current draw's at its slot and its
    climb bit, with every other classification field it records, is the producer's classifier (`bon_climb.classify_draw`)
    re-applied to that draw, under the plan's `classifier_fingerprint` = the tree's; every shard row's
    `response_512_sha256` is the store draw's at (goal_id, sample_idx, trial), at cap, redrawn once; the shards' slot set
    is the manifest's climb set); an unlocatable or unverifiable climb is refused by name with nothing claimed (never the
    floor under the climb's name), and a root is never recomputed. The judgment files the paper's climb set
    (`asr_bon_cap4096` at the judge cutoff and at 0.75, the counterfactual `asr_at_n{N}`, `bon_nauc`,
    `residual_at_cap_share_bon_4096`, `instrument_kind 512store_4096climb`), the one measured climb `run_degeneration` row,
    the item rows from the FOLDED texts, and the folded inputs (store sha256, climb root + locator, plan / manifest / every
    shard sha256) on the judgment's `extra.climb_inputs` and the headline's `extra.climb_inputs`; `grading_provenance.json`
    carries `registry.bon_climb_inputs`. The rung line and the dry-run plan name the climb root and its locator. The
    rebuild of item 6 never rebuilds a climb: it re-reads the store's cap-512 draws through the judge cache and has no fold
    step, so a located climb is refused by name before the rebuild considers its judgment (`bon_climb_rebuild_refusal`),
    and its cell is rebuilt only from another run of its store that passes.

Nothing else differs: the engine-identity guards, completeness gates, the judge pin for multi-turn and static legs
(`claude-haiku-4-5`, StrongREJECT v1, cache kind `sr`), `--dry-run`, `--goalset-dir`, `--cells`,
`--allow-errored-goals`, the aggregate schema and every log line are the source's. Of the imported modules, those on
the StrongREJECT grade path (`grading_live`, `throttle`, `proxy_engine`, `manifest`, `jailbreak_run`, `two_si_run`,
`grade_gated_served`, `jailbreak_artifacts`, `vault`) are identical to the source commit's except for the vault-mode change of
item 11 (`vault` gained the `wif` mode, op and env byte-identical; `grading_live` a prebuilt `client=` seam and the keyless stamp
on `pool_stamp`); `grade_bon_live` differs in the registry-side helpers this twin calls and the source never did (`bon_metric_rows`,
the rung helpers of items 9 and 14 and the climb fold helpers of item 15: BONHEADLINE, BONRUNG, BONFOLD) and in a
status line that prints either stamp shape, not on the StrongREJECT judge path.

## Retirement rule

The copy exists only because the source has not landed on `main`. When it does: grade from the landed file, file the
roster enumeration against it as an additive change, delete this copy, `GRADER_DIFF.patch` and this file, and retarget
`tests/test_seedrep_grader_twin.py`. A store already graded by the twin is not regraded on that basis alone:
`grading_provenance.json.lane_copy.source_sha256` names the bytes it was graded with; a grade re-runs only if the
landed file's diff against them touches a grade, a judge call or a row.
