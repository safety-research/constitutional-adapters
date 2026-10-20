# tinycap interfaces

The tinycap suite is the served capability battery (Inspect tasks and EvalPlus) run against a vLLM standup through the routing proxy. The machine-readable copy
of every contract is `src/sisteer/capeval/tinycap/contracts.py`: every name below is defined there,
and `tests/test_tinycap_contracts.py` fails if this document stops naming one. Import the names;
never retype a string. `contracts.py` does no scoring or dispatch, but beside the names it
carries the executable contract behaviour every component must agree on byte for byte: the
per-task thinking policy (`thinking_of_task`, `battery_thinking`, `check_thinking_arg`), identity
construction (`sampling_identity`, `container_name`, `cell_out_dir`, the hashed-field tuples),
parsing (`parse_launcher_cell`, `expand_control`, `parse_replace_source`,
`reduction_mode_from_env`, which takes the environment mapping as an argument) and the
attestation checks (`check_reduction_mode_expectation`, `sampling_for`). Those are pure functions
of their arguments. The exceptions are the named readers, which open the JSON / YAML document they
are handed and validate it (`read_row_engine_record`, `read_item_subset`, `read_oos_prompt_budget`,
`bcb_lib_families`, the subset-pin, IFEval-matrix and roster / bank-view document loaders), and
two helpers that read the process environment: `legacy_gsm8k_enabled` (`SISTEER_LEGACY_GSM8K`)
and `full22_roster_candidates` (`SISTEER_FULL22_ROSTER`, then the repository and the working
directory).

## 1. `--controls` grammar — `served_capability_run.py generate`

A control token is expanded by `contracts.expand_control(token)` into canonical **atoms**
(`CONTROL_ATOMS`); an unknown family, an unknown sub-task, or a sub-task on a family that takes
none raises `ControlNameError`.

| token | atoms | dispatch module |
|---|---|---|
| `math500` | `math500` | `sisteer.capeval.tiers.inspect_served` |
| `bfcl`, `hellaswag_inspect` | the same-named atom | same |
| `gsm8k` | `gsm8k` — the standard GSM8K: inspect_evals' stock task restricted to the 200 items matched by question text to the GSM8K-Platinum rows, row task `inspect/gsm8k_paired200`; fills the `gsm8k` control slot at exactly `GSM8K_PAIRED_N` = 200 items (a pinned-input drift refuses at plan time) | same |
| `gsm8k_full` | `gsm8k_full` — the full 1,319-item test set (`GSM8K_FULL_N`), row task `inspect/gsm8k` (`GSM8K_FULL_TASK`); never the slot | same |
| `inspect_cmp` | `inspect_cmp:mmlu`, `inspect_cmp:xstest`, `inspect_cmp:or_bench_hard_1k` | same |
| `inspect_cmp:gsm8k` | a retained spelling (`RETAINED_CONTROL_SPELLINGS`) resolved to the canonical `gsm8k` atom | same |
| `inspect_cmp:<task>` | that one atom | same |
| `humaneval_plus`, `mbpp_plus` | the same-named atom | `sisteer.capeval.tiers.evalplus_runner` + `serving/proxy_engine.py` |
| `bigcodebench_200` | inspect_evals' BigCodeBench (complete split, the 148 Hard ids excluded) narrowed to the 200 pinned ids of `configs/capeval/tasks/bcb_200.json`, row task `inspect/bigcodebench_200`, every sample executed in an owned Modal sandbox; opt-in | `inspect_served` |
| `bfcl_200` | the stock `bfcl` narrowed to the 200 pinned ids of `configs/capeval/tasks/bfcl_200.json`, row task `inspect/bfcl_200`; needs a `--tool-calls` standup; opt-in | same |
| `bigcodebench_full`, `bfcl_full` | the full BigCodeBench and the full single-turn BFCL as two more pinned legs; opt-in | same |
| `ifeval` | inspect_evals' stock `ifeval` task (541 prompts of `google/IFEval`), row task `inspect/ifeval`; k = 2 seeded trials, thinking off, pass-1 cap 2,048 with 4,096 / 8,192 rungs through `--off-cap-by-task`; opt-in, in no battery (§15) | same |

Related constants: `TINYCAP_CONTROLS`, `INSPECT_CMP_TASKS`,
`DISPATCH_MODULE_OF_CONTROL`, `TASKS_OF_CONTROL` (atom → row task names), `TOOL_CALL_CONTROLS`
(`bfcl`, `bfcl_200`: the families whose requests carry `tools`), `INSPECT_CONTROLS` /
`EVALPLUS_CONTROLS`, `EVALPLUS_DATASET_OF_CONTROL` (`humaneval_plus → humaneval`, `mbpp_plus → mbpp`).

**Batteries.** `STANDARD_BATTERY_PROMPTED` = `STANDARD_BATTERY_FITTED` = `gsm8k, humaneval_plus,
inspect_cmp:or_bench_hard_1k` — what a `--pass1` launch runs when no controls are named.
`IN_DEPTH_EXTRAS` = `math500, gsm8k_full` is what `--in-depth` appends
(`battery_controls(kind, in_depth=)`, kinds `BATTERY_PROMPTED` / `BATTERY_FITTED`). The
generative letter-choice legs (`inspect_cmp:mmlu`,
`hellaswag_inspect`), MBPP+ and BFCL are ordinary atoms that run only under an explicit
`--controls`. `--tool-calls` is derived from the requested controls, so no default standup
composes it: `needs_tool_calls(CONTROLS_PROMPTED) is False`.

**The GSM8K slot.** `GSM8K_SLOT_ACCEPTANCE` = `{task: inspect/gsm8k_paired200, item_set: paired200,
framework: inspect_evals-0.18.0, n: 200}`; `cell_controls.load_controls_summaries` fills the
`gsm8k` slot only from a payload carrying every one of those values
(`contracts.is_standard_gsm8k_slot_payload`). Any other `gsm8k` payload is returned under
`GSM8K_LEGACY_SLOT` = `gsm8k_legacy_s2parity`, retained and labelled, never inside the controls.
`GSM8K_PAIRED_TASK_NAME` = `gsm8k_paired200` is the unprefixed task name the cap flags use.

**The OR-Bench leg is a FROZEN, SEEDED 200-item subset** of the 1,319-prompt corpus, and the
whole corpus stays available behind the driver's `--or-bench-full` (`FLAG_OR_BENCH_FULL`) as a
different identity. The spec is pure and stdlib-only:
`or_bench_item_set` = `{item_set: "subset", n: OR_BENCH_SUBSET_N = 200, seed: OR_BENCH_SUBSET_SEED =
20260906, of: OR_BENCH_FULL_N = 1319, positions: <200 sorted indices>}`
(`contracts.or_bench_subset_positions` / `or_bench_item_set` / `run_or_bench_item_set`; the full
corpus is `{item_set: "full", n: 1319, seed: null, of: 1319, positions: null}`; a run that files no
OR-Bench rows records `null`). The draw is a SHA-256 ordering of `"<seed>:<position>"`, never
`random.sample`. The POSITIONS are hashed (`IDENTITY_KEY_OR_BENCH_ITEM_SET` = `or_bench_item_set`);
the resolved item ids ride, unhashed, in `manifest["or_bench_item_set"]`
(`MANIFEST_KEY_OR_BENCH_ITEM_SET`, `OR_BENCH_ITEM_SET_BLOCK_FIELDS` = `item_set, n, seed, of,
positions, item_ids`). `inspect_served.or_bench_selected_rows` re-checks the loaded corpus against
`OR_BENCH_FULL_N` at plan time and `or_bench_item_ids` is the resolution the manifest records.

**Per-task thinking policy.** One arm per row task, `THINKING_BY_TASK` (total over `TINYCAP_TASKS`):

| arm | row tasks |
|---|---|
| on | `inspect/math`, `inspect/gsm8k_paired200`, `inspect/gsm8k`, `inspect/mmlu_letter_valid`, `inspect/mmlu_0_shot`, `inspect/xstest` |
| off | `inspect/hellaswag`, `inspect/bfcl`, `inspect/or_bench_hard_1k`, `inspect/bigcodebench_200`, `inspect/bfcl_200`, `inspect/bigcodebench_full`, `inspect/bfcl_full`, `inspect/ifeval`, every `evalplus/*` task |

`THINKING_ONE_ARM_PER_TASK` is the rule: a task is never run under both arms on the same items.
`battery_thinking(kind, in_depth=)` is the `{row task: mode}` map over a battery. The driver's
`--thinking` takes `LAUNCHER_THINKING_MODES` = `off | on | per-task | not_available`
(`THINKING_PER_TASK` = `per-task`); `check_thinking_arg` validates it, `is_per_task` is the
predicate the driver branches on, and `run_thinking_by_task(atoms, mode)` is the run's
`{row task: arm}` map over the tasks it files — hashed as `IDENTITY_KEY_THINKING_BY_TASK` =
`thinking_by_task` in every mode. Under `per-task` an unmapped task refuses at plan time; the
provider resolves `chat_template_kwargs.enable_thinking` per request off the running task
(`TinycapAPI.request_thinking`), and the grader is built with no map and `thinking=off`. A
mixed-arm standup on a lens route is admissible only while the served template's assistant header
does not move between the arms: `measure_thinking_header_by_arm` compares the token ids of both
renders before anything is generated and refuses a lens route whose header moves; the measurement
lands in `evidence.json` as `thinking_header_by_arm` (`header_moves`, `header_len_by_arm`,
`header_ids_by_arm`, `template_comparison`), and `MixedArmHeaderMoved` / `HeaderMeasurementError`
are the typed refusals.

**MATH-500 cap: one pass at 16,384 in both arms.** `MATH500_MAX_TOKENS_DEFAULT` = 16,384
(`inspect_served.MAX_TOKENS_OFF` / `MAX_TOKENS_ON` read the one constant). There is no rung above.
The per-item retry is retired but re-armable: `MATH500_RETRY_MAX_TOKENS` = 0 is the driver's default
(`--math500-retry-max-tokens` = `FLAG_MATH500_RETRY_MAX_TOKENS`, not in `TINYCAP_DRIVER_FLAGS`); a
value above the pass cap re-requests every sample whose `stop_reason` is `STOP_REASON_AT_CAP` once,
with the same sample index, rows recording `attempt` and `max_tokens_used`. After attempt 1,
`assert_pass_fits_window` refuses the task by name when any cap-bound sample has
`prompt_tokens + max_tokens > max_model_len` (a sample stopped at the window would otherwise read as
cap-bound). Identity: `math500_retry_max_tokens` (`IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS`) is hashed on
every tinycap store, so a run that files no `inspect/math` row stamps the historical literal
`MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY` = 8192 (`contracts.math500_retry_identity_value(row_tasks,
retry)`), and a run that files MATH-500 also stamps `math500_max_tokens` (`IDENTITY_KEY_MATH500_MAX_TOKENS`),
the first-pass cap from `inspect_served.task_max_tokens` (the launcher's pure twin is
`math500_max_tokens_for`), present only when `files_math500` (`TINYCAP_IDENTITY_MATH500_FIELDS`
through `tinycap_identity_fields(files_math500=…)`). Every non-MATH identity is pinned byte for byte by
the golden fixtures under `tests/fixtures/tinycap_identity_v1/`. MATH-500 serves at its own geometry
class `gen_long_math` (§5): `capability:gen_long_math` at 17,664 and `capability:gen_long_math_longsi`
at 24,064.

**Sampling contract.** The generative evals decode as k seeded samples per item at each model's card
temperature, frozen per model with `top_p` / `top_k`; k = 1 on every task except IFEval (k = 2).

* **Modes** — `SAMPLING_MODES` = `greedy | card | thinking_card | general | battery | protocol |
  vendor_preset`, the driver flag `--sampling` (`FLAG_SAMPLING`), default `SAMPLING_DEFAULT` = `card`.
  `thinking_card` (`THINKING_SAMPLING_BY_MODEL_KEY`), `general` (`GENERAL_SAMPLING_BY_MODEL_KEY`,
  qwen35_4b only) and `battery` (`BATTERY_SAMPLING_BY_MODEL_KEY`) each have their own literal table;
  `protocol` is §14's per-task reference decoding. `greedy` is argmax, one sample per item, no
  nucleus keys (`GREEDY_SAMPLING`) and a different identity. Under `card`, k = 1 is a
  SINGLE STOCHASTIC SAMPLE at the model's card row — a different instrument from `greedy`'s argmax; the two
  never share a store key. `SamplingRow(temperature, top_p, top_k, source, min_p, presence_penalty,
  repetition_penalty)`; the optional members are hashed only when set
  (`SAMPLING_OPTIONAL_IDENTITY_FIELDS`).
* **The per-model table** — `SAMPLING_BY_MODEL_KEY: {model_key: SamplingRow}` over `geometry.MODEL_KEYS`
  (`llama8b`, `llama3b`, `gemma4e4b`, `gemma4e2b`, `qwen35_9b`, `qwen35_4b`, `glm46vflash`,
  `nemotron3nano4b`); `sampling_for(model_key, mode)` resolves a run's row and refuses
  (`SamplingUnverified`) a key with no row — never a default temperature. The `source` prose is not
  hashed; the three numbers are.

  | model key | temperature | top_p | top_k | source |
  |---|---|---|---|---|
  | `gemma4e2b`, `gemma4e4b` | 1.0 | 0.95 | 64 | the E2B snapshot's `generation_config.json` (E4B ships none and inherits it) |
  | `llama8b`, `llama3b` | 0.6 | 0.9 | — | Llama-3.1-8B-Instruct's `generation_config.json` (3B inherits) |
  | `qwen35_9b`, `qwen35_4b` | 0.7 | 0.95 | 20 | the Qwen3.5 model card, non-thinking mode |
  | `glm46vflash` | 0.8 | 0.6 | 2 | GLM-4.6V-Flash's `generation_config.json` |
  | `nemotron3nano4b` | — | — | — | no row (`SAMPLING_UNVERIFIED_MODEL_KEYS`): the card recommendation is unverified; the refusal records what the snapshot ships |

* **The server's default sampling** for a model that ships no `generation_config.json` —
  `SERVER_DEFAULT_SAMPLING_BY_MODEL` / `server_default_generation_config(row)`, composed by
  `vllm_server.build_server_launch` as `--override-generation-config` (the Qwen3.5 vendor rows are
  frozen in `QWEN35_VENDOR_ROWS`; only `SERVER_DEFAULTABLE_SAMPLING_KEYS` can ride this way).
* **Samples per item** — `SAMPLES_BY_TASK` (total over `TINYCAP_TASKS`): every task is a decoding
  task (`DECODING_TASKS` = every `inspect/*` and `evalplus/*` task) and takes
  `DECODE_SAMPLES_PER_ITEM` = 1 and MATH-500 `MATH500_SAMPLES_PER_ITEM` = 1; `inspect/ifeval` is
  `IFEVAL_SAMPLES_PER_ITEM` = 2 (§15). `samples_of_task(mode, task, explicit_samples=)` is one
  task's k (`math_samples=` is the same knob's older alias; naming both refuses) and
  `run_samples_by_task(tasks, mode, explicit_samples=)` the sorted `{row task: k}` map over the
  tasks a run files. The driver's `--samples N` is an explicit k for the two tasks in
  `SAMPLES_FLAG_TASKS` — MATH-500 and IFEval — in either sampling mode (`greedy` included); every
  other task ignores it. Left unset, each task takes its default k above (1 under `greedy`). The
  effective map is the identity's `samples_by_task`, so an explicit value is hashed into the store
  identity and recorded in the manifest.
* **Seed rules** — `SEED_RULES` = `{inspect: SEED_RULE_INSPECT, evalplus: SEED_RULE_EVALPLUS}`:
  `sha256(item_id NUL sample_idx)[:4] big-endian mod 2^31-1` (`contracts.sample_seed`) and
  `blake2b(task, item_id, sample_idx) mod 2^31-1` (`proxy_engine.stable_seed`). Reproducible in
  distribution only (bf16 batch composition).
* **Identity** — `SAMPLING_IDENTITY_FIELDS` = `temperature, top_p, top_k, samples_by_task, seed_rule`,
  the tail of `TINYCAP_IDENTITY_FIELDS` (`sampling_identity(row, samples_by_task)`); mirrored with
  `mode`, `model_key` and `source` in the manifest's `sampling` block (`MANIFEST_KEY_SAMPLING`,
  `SAMPLING_BLOCK_FIELDS`), which the resume guard `_check_manifest_sampling` reads.
* **No forks.** k samples are k separate requests; every leg refuses `n`
  (`proxy_engine.assert_client_body`), and the proxy refuses `n` / `best_of` above 1 on a steered route
  (`proxy.SteeredForkRefusal`, code `steered_fork_unsupported`).
* **Pairing.** `tiers/pairing.py` reduces avg@k inputs to the per-item mean FIRST, then pairs
  (`_reduce_item`); `paired_bootstrap` resamples items, never samples; `samples_summary` reports each
  side's min / max samples per item.
* **EvalPlus under k samples** — `render_chat_items(samples_per_item=)` renders k items per problem,
  the engine files each generation under `proxy_engine.generation_key(item_id, sample_idx)` =
  `<item_id>/<sample_idx>`, the sandbox reads k off `gen_meta` (`samples_per_item_of`), and the
  converter files per-sample `acc@<i>` rows beside per-item `pass@1_base` / `pass@1_plus` / `acc` rows
  carrying the unbiased estimator `pass_at_k_unbiased(n, c, 1)` = c / n (named `PASS_AT_1_ESTIMATOR`
  in their `extra_json`). k = 1 is a draw too: a store generated under the contract files an `acc@0`
  row with the generation's `seed`, `temperature`, `top_p`, `top_k`.

**Driver flags** (`TINYCAP_DRIVER_FLAGS`; the launcher passes them through `contracts.driver_argv`):

| flag | meaning |
|---|---|
| `--thinking on\|off\|per-task` | required whenever a tinycap token is requested; travels on every request as `chat_template_kwargs: {"enable_thinking": b}` |
| `--samples N` | explicit samples per item for MATH-500 and IFEval (`SAMPLES_FLAG_TASKS`; defaults 1 and 2); other tasks ignore it. N independent requests with distinct seeds, never `n=`; the value is hashed into the identity's `samples_by_task` |
| `--inspect-limit N` | smoke cap on Inspect tasks (0 = full) |
| `--ledger-url URL` | the usage ledger's `/v1` base; all tinycap traffic goes through it. The ledger's record `ledger_endpoint.json` (`LEDGER_ENDPOINT_FILENAME`) must exist and name this URL and the standup's proxy as `upstream`, else `EndpointBindingMismatch` |
| `--out-root DIR` | the launcher points it at `cell_out_dir(root, cell, mode)` = `<root>/<model>/<cell_id>/thinking_<mode>` |

Before any served fact is copied off the standup's `endpoint.json`, the driver binds the record to
the endpoints it was pointed at (`endpoint.proxy_url` = `--proxy-url`, `endpoint.routing_table` =
`--routing-table`, the ledger record's URL and `upstream`); any disagreement is
`EndpointBindingMismatch`. The verified endpoints are recorded in the manifest's `binding` block.

Standup flags per configuration (`contracts.standup_extra_argv`): `--tool-calls` on every BFCL
standup; `--reasoning-parser <family>` (`FLAG_REASONING_PARSER`) under thinking-ON, pinned by
`REASONING_PARSER_BY_FAMILY`. The standup's `--petri-agent-loop` (default on; `--no-petri-agent-loop`
= `FLAG_NO_PETRI_AGENT_LOOP`) is recorded, not composed: `endpoint.json` carries `agent_loop_entry`
(`AGENT_LOOP_ENTRY_KEY`, required) and `lens_reduction_mode` (`PROXY_IDENTITY_REDUCTION_MODE`), the
latter read from every worker's steering attestation.

**The lens reduction mode** (`REDUCTION_MODE_FIELD` = `reduction_mode`; `REDUCTION_MODES` = `per_row`,
`stacked`) has three readings in a standup record: the key absent is a pre-landing server, read as
`per_row`; a stated mode is that mode; the key null with a `lens_reduction_mode_error` refuses
(`REDUCTION_MODE_UNATTESTED`) rather than defaulting. `contracts.resolve_reduction_mode` applies the
rule; `endpoint_reduction_mode` is the launcher's torch-free copy of `lens_worker.endpoint_reduction_mode`.
The standup's switch is `SISTEER_LENS_STACKED_REDUCTIONS` (`LENS_STACKED_REDUCTIONS_ENV`), parsed by
`contracts.reduction_mode_from_env` (`REDUCTION_MODE_ENV_VALUES`: unset / `1` / `stacked` → `stacked`;
`0` / `per_row` / `off` → `per_row`; anything else refuses). The launcher asks for the mode explicitly
(`--reduction-mode {per_row,stacked}`; `REDUCTION_MODE_LAUNCHER_DEFAULT` = `per_row`; `reduction_mode_expected`)
and `check_reduction_mode_expectation` holds that expectation to the attested value — a disagreement
is `REDUCTION_MODE_MISMATCH`, and an expectation of `stacked` against a keyless server refuses too.

**BFCL on steered (lens) cells.** The single-turn BFCL categories send ordinary tail renders the
lens serves undeclared; a tool TURN is an agent-loop shape the proxy's Petri-facing entry declares.
`inspect_served.assert_route_can_serve(names, route, agent_loop_entry=, tool_calls=)` refuses BFCL
at plan time unless the standup composed `--tool-calls` and, on a lens route, declared the entry;
`assert_live_agent_loop_entry` reads the live `/v1/models` identity block
(`petri_proxy.petri_agent_loop_entry` = `PROXY_IDENTITY_AGENT_LOOP_ENTRY`) and requires it to equal
the record. `agent_loop_entry_of` is the launcher's reading of the flag, and
`_check_live_proxy_identity` holds the live block to the record after the proxy client loads.

## 2. Rows (`ROW_COLUMNS` frozen — new facts ride in `extra_json`)

All tinycap rows: `tier = 1`, `engine = "vllm"`, aggregates at `item_id = "__aggregate__"`.
Completion identity is `(task, item_id, metric)` (`CellStore.completed_keys`); a task with a DONE
marker is skipped whole by a resume and its persisted `summary/<task>.json` returned.

| family | `task` | `task_version` | `item_id` | per-item metrics | `extra_json` |
|---|---|---|---|---|---|
| Inspect | `inspect/<name>` | `inspect_evals-0.18.0` (`INSPECT_TASK_VERSION`) | Inspect's sample id (`inspect_sample_id(prefix, *fields)` = `<prefix>_<md5("\0".join(fields))[:8]>`) | `acc` (per-item mean); per-sample `acc@<k>` (`sample_metric("acc", k)`) when k > 1 | `sample_idx, seed, answer, prompt_tokens, completion_tokens, paired_id, stop_reason, unparseable, attempt, max_tokens_used` — per sample |
| EvalPlus | `evalplus/humaneval_plus`, `evalplus/mbpp_plus`, + `:open` variants | `evalplus-0.3.1+base_and_plus` (`evalplus_row_task_version(metric_def)`; the framework pin `EVALPLUS_TASK_VERSION` = `evalplus-0.3.1` plus the definition) | `HumanEval/<n>` / `Mbpp/<n>` | `pass@1_base`, `pass@1_plus` (base AND plus tests), `acc` (= plus); per-sample `acc@<i>` | `metric_def` on every `pass@1_plus` / `acc` row; `samples_per_item`, `estimator`, `values`, `n_correct` when k > 1 |

The row task names, exactly (`TINYCAP_TASKS`): `inspect/math`,
`inspect/bfcl`, `inspect/hellaswag`, `inspect/mmlu_letter_valid`, `inspect/mmlu_0_shot`,
`inspect/gsm8k`, `inspect/gsm8k_paired200`, `inspect/xstest`, `inspect/or_bench_hard_1k`,
`inspect/bigcodebench_200`, `inspect/bfcl_200`, `inspect/ifeval`, `inspect/bigcodebench_full`,
`inspect/bfcl_full`; `evalplus/humaneval_plus`, `evalplus/humaneval_plus:open`, `evalplus/mbpp_plus`,
`evalplus/mbpp_plus:open`. Domains (`TINYCAP_DOMAINS`, `TINYCAP_DOMAIN_OF[task]`): `inspect_gen`,
`evalplus_served`. `configs/capeval/domains.yaml` also keeps the `served_ll` domain of the removed
loglikelihood family, so rows earlier stores filed under it still resolve.

**Producer facts on every Inspect row** (`INSPECT_EXTRA_FIELDS`): `stop_reason` (inspect_ai's
`StopReason` verbatim; `max_tokens` is at-cap, `inspect_served.STOP_REASON_AT_CAP`; `model_length` is a
context overflow and is not at-cap), `unparseable` (`True` when the scorer recorded an empty answer,
`null` when it records no answer at all), `attempt` (1, or 2 for a retried MATH-500 sample),
`max_tokens_used` (the cap the generation was requested at). Each task's summary carries the two rates
with their denominators: `unparseable_rate` = `n_unparseable / n_answered` and `at_cap_rate` =
`n_at_cap / n_stop_reason`; either is `null` when its denominator is 0 (`inspect_served.task_quality`).
An unscored result (inspect's NaN sentinel) is recorded as `extra_json.unscored: true`, left out of
every mean, and counted in `n_unscored`; `acc` is the mean over `n_scored`. At k > 1 the per-item row's
singular facts come from the first scored draw (`provenance_sample_index`) and `per_sample` carries
every draw's compact provenance.

**The `pass@1_plus` definition is named.** `extra_json.metric_def` = `base_and_plus` on every
`pass@1_plus` and `acc` row and in the summary payload (`EVALPLUS_METRIC_DEF_FIELD`; `EVALPLUS_PLUS_DEFS` =
`base_and_plus` | `plus_only`). A row or payload without the field was written under the old
plus-only reading: `load_controls_summaries` stamps it `plus_only` at load, and `metric_def_mismatch`
refuses to pair two payloads whose definitions differ. The definition is identity-bearing on the rows
through `task_version`, so the converter validates the existing rows of a task before skipping any
(`CellStore.rows_of_task` → `evalplus_runner._check_existing_rows_definition`): a row under another
`task_version`, or whose `metric_def` disagrees with the definition this tree writes, refuses the
conversion; the remedy is `scripts/tinycap_rederive_evalplus.py <cell_dir>… --out <dir>`, which
recomputes the rows off-volume from the retained `eval_results.json` into a sidecar. The version a
consumer pools on is the EFFECTIVE one, `noninferiority.effective_task_version(df, task, what)` →
`contracts.evalplus_effective_task_version(task_version, metric_defs)`: bare + every row `base_and_plus`
is `evalplus-0.3.1+base_and_plus`; bare + absent is plus-only; a MIXED or contradictory group refuses.
Every EvalPlus payload the summary writer compares or writes is normalised through
`contracts.with_evalplus_metric_def` (absent = `plus_only`), so a metric-definition change
is never an ordinary supersession (§3).

**Completion state** (`sisteer.capeval.tinycap.completion`; `completion.finish_task`; marker files via
`marker_filename(task)`):

| file | when | fields |
|---|---|---|
| `<cell_dir>/done/<task>.json` | the run was unlimited and the scored set equals the frozen full count | `task, n_items, n_rows, n_requests, n_rejected, wall_clock_s, finished_at` (`DONE_MARKER_FIELDS`) + at most `samples_sha256` and `rebound_at` (`DONE_MARKER_REBOUND_AT_FIELD`, written only by the sandbox's re-bind mode, §9) |
| `<cell_dir>/partial/<task>.json` (`PARTIAL_DIR`) | any limit active, or the set short of the full count | the done fields + `n_expected_full` + `limits` (`PARTIAL_MARKER_FIELDS` = `task, n_items, n_expected_full, limits, n_rows, n_requests, n_rejected, wall_clock_s, finished_at`) |
| `<cell_dir>/summary/<task>.json` (`SUMMARY_DIR`) | every finished task | the task's counts-only summary payload |

An item that produced no generation refuses its task (naming the item, the draw, the HTTP status, and
how many sibling samples were merely cancelled when the eval tore down); `fail_on_error` is always on.
`n == n_scored + n_unscored` over a full attempted set.

### Deferred: errored-item completion

There is no path that completes a task holding items that produced no generation. A re-introduction
owes four things: (1) a DRAW-level invariant, `n * samples_per_item == n_scored_samples +
n_unscored_samples + n_errored_samples`, not the item-level one, which cannot see a lost draw; (2) a
non-zero `n_errored_samples` never produces a fixed-k DONE artifact (refuse, end PARTIAL, or publish
under a different metric and identity); (3) error-PARTIAL and limit-PARTIAL need different transition
semantics, and the DONE marker must atomically replace the PARTIAL one; (4) the k = 2 split-outcome
case (first draw errors, second scores) is pinned by a test. Today the k = 2 case is pinned as a
refusal (`tests/test_inspect_served.py`). The usage ledger (§6) counts requests, not items, so it
cannot say which item was lost.

## 3. Cell `manifest.json` — `TINYCAP_MANIFEST_KEYS`

| key | value |
|---|---|
| `thinking` | `{enable_thinking, render, reasoning_parser, mode, by_task}` (`THINKING_BLOCK_FIELDS`); `enable_thinking` is null under `per-task`; `_check_manifest_thinking_map` is the resume guard |
| `geometry` | the standup's `endpoint.json` `geometry` block + `pass` + `source`; not identity (a resume rewrites it) — the hashed `tinycap.pass` is the cross-check |
| `concurrency` | the resolved S the client fanned out at |
| `route` | `{virtual_model, grader_virtual_model, serving_identity, tool_calls, agent_loop_entry, reduction_mode}` (`ROUTE_BLOCK_FIELDS`); the grader is the routing table's `none` entry; a manifest without `reduction_mode` is legacy `per_row` |
| `declaration` | `{response_prefill_used: [tasks]}` |
| `seeds` | the seed rule + constants per requested family; the per-(item, sample) values ride in every row's `extra_json` |
| `frameworks` | `{inspect_ai, inspect_evals, evalplus, client_lock_sha256}` — the imported version of every requested framework, held to `FRAMEWORK_PINS` = `{inspect_ai: 0.3.260, inspect_evals: 0.18.0, evalplus: 0.3.1}` at startup (`FrameworkPinMismatch`); hashed identity |
| `usage_ledger` | the durable path `<out_root>/usage_ledger.jsonl` when `--ledger-url` was given |
| `sandbox` | `{image_digest, cpu, memory_mib, timeout_s, block_network}` when an EvalPlus family is requested |
| `tinycap` | mirrors `resolved_config["tinycap"]` |
| `sampling` | `{mode, model_key, temperature, top_p, top_k, samples_by_task, seed_rule, source}` (`SAMPLING_BLOCK_FIELDS`); under `protocol` also `by_task` (`SAMPLING_BLOCK_KEY_BY_TASK`, §14) |

Top-level, driver-named and unhashed: `binding` (the verified endpoints), `producer`
(`MANIFEST_KEY_PRODUCER`; `PRODUCER_BLOCK_FIELDS` = `tree_sha, driver` — `tree_sha` is the content of the
`TREE_SHA` file (`TREE_SHA_FILENAME`) at the tree's root, else `git rev-parse HEAD`; an immutable resume
key through `CellStore.CONDITIONAL_RESUME_KEYS`, compared whenever the existing manifest carries it),
`provenance` (`MANIFEST_KEY_PROVENANCE`, `cell_controls.controls_provenance(...)` with the served stamp),
`quality` (`MANIFEST_KEY_QUALITY`; `QUALITY_BLOCK_FIELDS` = `unparseable, at_cap, by_task` — the run's
per-cell roll-up of the two rates, pooled from counts, written after generation), `or_bench_item_set`
(§1), `item_subset`, `on_cap_by_task`, `attempt_attribution`, `accepted_generation_attempt` (§13) and
`row_engine` (§16).

**Identity.** `resolved_config["tinycap"]` carries `TINYCAP_IDENTITY_FIELDS` = `tasks, thinking,
versions, served, limits, samples, frameworks, pass, thinking_by_task, math500_retry_max_tokens,
or_bench_item_set, temperature, top_p, top_k, samples_by_task, seed_rule` (+ `math500_max_tokens` on a
MATH run, `max_tokens_by_task` under §11's comparator, `sampling_by_task` under §14's protocol mode,
`cap_rung` under §13's rung store); `served` = `SERVED_IDENTITY_FIELDS` = `reasoning_parser, tool_calls,
max_model_len, max_num_seqs, concurrency, agent_loop_entry, reduction_mode`, read off `endpoint.json`
and the workers' attestation. A limited store, another served parser / tool-call mode / window / S /
client width / framework environment / geometry pass / thinking map / cap / draw never shares a cell
key with the full run; a colliding key is additionally refused by `_check_manifest_serving_identity`
(`ServingIdentityMismatch`), `_check_manifest_thinking_map` and `_check_manifest_sampling`. The producing
tree is compared by the launcher first: `untrusted_store_reason` requires `producer.tree_sha` to equal
the staged tree's stamp, and `producer_tree_conflicts` returns `PRODUCER_TREE_MISMATCH` before any standup.

**The one summary writer:** `sisteer.capeval.tinycap.completion.merge_controls_summary(cell_dir)`
(`CONTROLS_SUMMARY_MERGE_MODULE` / `CONTROLS_SUMMARY_MERGE_FUNCTION`) collects every task with a DONE
marker and merges its payload into `<out_root>/<cell_id>/controls_summary.json` through
`cell_controls.write_controls_summary` (read-modify-write, provenance equality asserted, atomic
replace). Payload keys (`SUMMARY_PAYLOAD_FIELDS`, counts only): `acc, acc_norm, irt_gpirt, n, n_requests,
n_rejected, samples_per_item, pass_at_1_base, pass_at_1_plus, framework, engine, route, thinking,
wall_clock_s, n_scored, n_unscored` (+ a family's further counts). Each Inspect payload names its `task`
and `item_set`; `completion.merge_controls_summary` also fills the control slot a done task maps to
(`SLOT_OF_TASK`: `inspect/gsm8k_paired200` → `gsm8k`, projected onto `GSM8K_SLOT_PAYLOAD_FIELDS`).

**Per-entry merge provenance.** Every tinycap entry the merge writes carries its source in the sibling
block `sources` (`SUMMARY_KEY_SOURCES`), keyed like the entries:

```json
"sources": {"inspect/gsm8k": {"cell_key": "…", "harness_config_hash": "…", "pass": 2,
                              "producer_tree_sha": "…", "payload_digest": "<16 hex>",
                              "superseded": [{"cell_key": "…", "harness_config_hash": "…", "pass": 1,
                                              "producer_tree_sha": "…", "payload_digest": "…"}],
                              "refreshed": [{"payload_digest_was": "…", "payload_digest": "…", "writer_tree_sha": "…"}],
                              "rebound": [{"at": "…", "payload_digest_was": "…", "payload_digest": "…",
                                           "samples_sha256": "…", "witnesses_agreed": ["…"], "writer_tree_sha": "…"}]}}
```

| name | meaning |
|---|---|
| `MERGE_SOURCE_RECORD_FIELDS` = `cell_key, harness_config_hash, pass, producer_tree_sha, superseded` | the four identity facts of the store an entry stands with, plus its history |
| `requires_merge_source(key, payload)` | which entries need a source: a tinycap row task, or a control slot whose payload names one |
| `UNRECORDED_MERGE_SOURCE` | the all-null record stamped on an entry a pre-discipline writer filed; a missing source therefore means torn |
| `MERGE_SOURCE_PASS_SOURCE_FIELD` = `pass_source`, `PASS_SOURCE_LEGACY_TOP_LEVEL` = `legacy_top_level` | the pass was read off the mutable top-level `geometry.pass`; recorded, but such a pass never orders a supersession |
| `MERGE_SOURCE_WRITER_TREE_SHA_FIELD` = `writer_tree_sha` | the tree that wrote the record in the launcher's two-tree merge (`writer_tree_sha_of(current_tree_src())`) |
| `MERGE_SOURCE_EQUAL_WITNESSES_FIELD` = `equal_witnesses`, `MERGE_SOURCE_EQUAL_WITNESS_RULE_FIELD` = `equal_witness_rule`, `EQUAL_WITNESS_RULE` = `hashed_pass_then_smallest_cell_key` | when several stores exactly witness one unrecorded standing payload, `completion.equal_witness_plan` picks the adopter by this rule before the first write |
| `MERGE_SOURCE_PAYLOAD_DIGEST_FIELD` = `payload_digest`, `merge_payload_digest(key, payload)` | sorted-key compact JSON of the normalised payload, sha256, 16 hex; a settled decision holds only for the payload it was made on, and a record without a digest settles nothing |
| `MERGE_SOURCE_REFRESHED_FIELD` = `refreshed`, `MERGE_REFRESH_DIGEST_WAS_FIELD` = `payload_digest_was`, `MERGE_REFRESH_RECORD_FIELDS` = `payload_digest_was, payload_digest, writer_tree_sha` | the standing source's own changed summary is adopted only when `--replace-source` names the standing store itself; the writer says `REFRESHED` |
| `MERGE_SOURCE_WRITER_FIELDS` = `payload_digest, refreshed, samples_sha256, samples_binding, execution_producer, rebound` | writer-owned fields a caller's source facts may not carry; the standing digest lives under `payload_digest`, never a separate `standing_digest` |
| `MERGE_SOURCE_REBOUND_FIELD` = `rebound`, `MERGE_REBIND_RECORD_FIELDS` = `at, payload_digest_was, payload_digest, samples_sha256, witnesses_agreed, writer_tree_sha` | the sandbox's re-bind of an EvalPlus entry's samples digest (§9); the writer says `RE-BOUND` |
| `MERGE_PAYLOAD_FIELDS` = `cell_dir, payloads`, `MERGE_EMIT_ATTEST_KEY` = `__completion__` | the scratch file the producing tree emits in the two-tree merge, and the key under which each half attests the `completion.__file__` it imported |

The rule when a second store claims an entry: the higher geometry pass wins and records the other as
`superseded`; a lower pass leaves the standing entry alone; the identical source re-merges
byte-identically; a source already recorded as superseded re-offers nothing; anything else (two stores
at the same pass, a `legacy_top_level` pass on either side, a differing payload over an unrecorded
source) refuses until the operator names the winner per entry with `--replace-source
<entry>=<cell_key>[,…]` (`parse_replace_source`; an entry must be one of `replace_source_entries()`).
An unrecorded standing payload equal to the offered one adopts the offered store as its source. A
metric-definition change refuses before adoption in either pass direction (`metric_def_mismatch`).
`cell_controls.load_controls_sources(out_root, registry, cfg)` returns every tinycap entry's source
record beside the number.

**Where the merge runs.** The served driver merges its own store in-process. The EvalPlus tasks
finish in the sandbox orchestrator, so their merge runs in the launcher's client venv
(`tinycap_served.merge_summaries`, `modal run … --merge`; the orchestrator chains it with
`--merge-mode merge` or prints the commands under `--merge-mode no-merge`) in two subprocesses per
group of stores: the PRODUCING tree only emits each store's normalised payload to a scratch file, then
the CURRENT tree performs the one sourced atomic write per store through
`completion.merge_emitted_payload(cell_dir, payload, replace_source=, writer_tree_sha=, plan=)`. Both run
under `python -P` with `PYTHONSAFEPATH=1` from an empty working directory, and the writer's stdout
lines (`ADOPTED`, `KEPT from`, `REFRESHED`, …) come back as `notes`. The writer also stamps
`reduction_mode` onto every offered payload from the source store's manifest `route.reduction_mode`
(`per_row` spelled by absence), and the loader refuses an entry that contradicts its store. The thin
Inspect MMLU task is built with `PARITY_N` = 200 rows. A `partial/` store is never merged.

## 4. Scorer — `sisteer.capeval.serving.prefill_logprob`

```python
PromptLogprobs = list[dict[int, float] | None]
class PrefillLogprobError(RuntimeError)
@dataclass(frozen=True) class ContinuationSpan(start: int, stop: int, header_at: int)   # .positions -> range
@dataclass(frozen=True) class CandidateScore(ll_sum: float, n_cont_tokens: int, n_chars: int,
                                             span_start: int, prompt_tokens: int | None, token_ids: tuple[int, ...])

build_prefill_request(*, virtual_model, user_text, candidate, system, response_prefill: bool,
                      enable_thinking: bool, seed: int = 0, thought_block_diag: bool = False,
                      return_token_ids: bool = True) -> dict
build_tokenize_request(*, virtual_model, messages, enable_thinking: bool) -> dict
normalize_prompt_logprobs(resp_json, *, expected_len: int | None = None) -> PromptLogprobs
prompt_token_ids_of(resp_json) -> list[int] | None
continuation_span(ids, header_ids, *, expected_candidate_ids=None) -> ContinuationSpan
continuation_logprob(plp, ids, span) -> float
score_candidate(*, resp_json, tokenize_ids, header_ids, n_chars, expected_candidate_ids=None) -> CandidateScore
```

`prompt_logprobs` over JSON is a list aligned with prompt positions, `null` at position 0, else an
object keyed by token id as a string with `{"logprob", "rank", "decoded_token"}`.
`normalize_prompt_logprobs` refuses a non-list, a wrong length, a non-base-10 key, a
missing / non-numeric / non-finite `logprob`, a duplicate key; `score_candidate` additionally refuses a
`prompt_token_ids` echo that differs from the tokenize ids. Request keys ⊆ `PREFILL_REQUEST_KEYS`;
tokenize keys == `TOKENIZE_REQUEST_KEYS`. The three-legged steering control runs `BITES_LEGS` =
`steered, scale0, scale0_repeat` on `BITES_PROBE_ITEMS` probes and is `applicable: false` on the
prompted route.

## 5. Geometry sub-entries

`GEOMETRY_SUBENTRY_KEYS`, exactly eight: `capability:tiny`, `capability:tiny_prompted`,
`capability:gen_long`, `capability:gen_long_thinking`, `capability:agent`, `capability:agent_thinking`,
`capability:gen_long_math`, `capability:gen_long_math_longsi`. Class per control family
(`GEOMETRY_CLASS_OF_CONTROL`): `humaneval_plus` / `mbpp_plus` / `hellaswag_inspect` /
`gsm8k` / `gsm8k_full` / `inspect_cmp` → `gen_long`; `math500` → `gen_long_math`
(`GEOMETRY_CLASS_GEN_LONG_MATH`); `bfcl` → `agent`. `geometry_subentry(cls, prompted=, thinking=, si=)`
applies `_prompted` (tiny only) and `_thinking` (gen_long / agent only; never `gen_long_math`).
No control pools into `tiny` since the loglikelihood family was removed; its two entries stay
registered as the measured windows earlier stores ran at.

| key | L × S (certified budget) | basis |
|---|---|---|
| `capability:tiny` | 1,792 × 219 | measured, p99_total |
| `capability:tiny_prompted` | 8,448 × 46 | provisional, analytic (sized from the longest SI); refuses pass 2 |
| `capability:gen_long` | 4,608 × 85 | measured, p99_prompt + max_tokens |
| `capability:gen_long_thinking` | 10,752 × 36 | measured, p99_prompt + max_tokens |
| `capability:agent` | 3,328 × 118 | measured |
| `capability:agent_thinking` | 6,400 × 61 | measured |
| `capability:gen_long_math` | 17,664 × 22 | the MATH-500 prompt max 1,041 + 16,384, rounded to 256; `max_prompt_plus_max_tokens` |
| `capability:gen_long_math_longsi` | 24,064 × 16 | the long-SI prompt max 7,622 + 16,384; `max_prompt_plus_max_tokens` |

MATH-500's window is sized to the prompt MAX, not the p99, because `assert_pass_fits_window` refuses a
task when any cap-bound sample has prompt + cap > L; the clamp at L is 0 by construction. Its window
is chosen by the PROMPT CLASS of the cell's system turn: `contracts.MATH500_PROMPT_CLASS_OF_SI` =
`{safety_refusal: short, si_constitution: long_si, si_irrelevant: long_si}` (no system turn → `short`),
`MATH500_GEOMETRY_KEY_OF_CLASS` the class → entry map; `assert_math500_class_fits` holds the resolved
window to that class's measured prompt max + the cap on every resolution, and an SI in neither class
refuses by name at plan time (`Math500PromptClassUnmeasured`). The model-grader traffic pools into the
informational key `GEOMETRY_GRADER_KEY` = `capability:grader`, refused by `resolve_geometry`. Pass 1 runs
every configuration at the certified Petri window with S from the served model's measured budget
(`sg.pass1_geometry(key, model_key=)`; an unmeasured budget refuses, `UnmeasuredBudget`); a MATH-only
configuration resolves `gen_long_math` in pass 1 as in pass 2, and MATH-500 pooled with other
sub-entries is refused (`Pass1FootprintCannotHold`). Pass 2 serves the measured window through
`resolve_geometry(key, model_key=)`. The measured rows and their derivation are in
`docs/served_geometry_p99.md` §6.

## 6. Usage ledger — `usage_ledger.jsonl`

One flat row per attempted `/v1/chat/completions` call (`LEDGER_ROW_FIELDS`): `ts, task, sample, vmodel,
thinking, prompt_tokens_pre, max_tokens_requested, has_tools, n_tools, status, error_code, rejected,
usage_prompt_tokens, usage_completion_tokens, finish_reason, latency_s, tokenize_mismatch, error_class,
request_id, body_sha256`. `prompt_tokens_pre` = `/v1/tokenize` of the exact outbound body before
forwarding. `error_class` (`LEDGER_ERROR_CLASSES`) is derived in the ledger: `none`, `context_length`,
`bad_request_other`, `auth`, `route_refused`, `rate_limited`, `server_error`, `transport`,
`ledger_blocked`; the geometry measurer scores a rejection at prompt + requested cap only under
`context_length`. Headers every client sets, consumed by the ledger and never forwarded
(`TINYCAP_HEADERS`): `X-Tinycap-Task: <row task>`, `X-Tinycap-Sample: <item_id>/<sample_idx>`
(`sample_header_value`). The ledger's endpoint record `{ledger_url, upstream, ledger_path, pid, probe}`
is written as `ledger_endpoint.json` (`LEDGER_ENDPOINT_FILENAME`) beside the standup's `routing_table.json`.

## 7. Inspect provider

`@modelapi(name="tinycap")` (`INSPECT_PROVIDER_NAME`); model string `tinycap/<vmodel>`
(`inspect_model_name`); env `TINYCAP_BASE_URL` (`INSPECT_BASE_URL_ENV`, the ledger's `/v1`) and
`TINYCAP_API_KEY` (`INSPECT_API_KEY_ENV`, the proxy bearer). `max_retries=2`, `timeout=7200`,
non-streaming, never `n`. The provider is route-bound (`build_model(..., route=)`) and verifies every
completion's four attestation stamps (`model`, `petri_intervention_hash`, `petri.cell_id`,
`petri.serving_identity`) with `proxy_engine.assert_response_stamps`. The thinking flag and the
sampling row are per-request facts (`build_model(..., thinking_by_task=, sampling=)`;
`TinycapAPI.request_thinking`, `TinycapAPI.sampling_params`); the grader is always built with
`GREEDY_SAMPLING` and `thinking=off`. `MODEL_GRADED_INSPECT_TASKS` = `inspect/{math, xstest,
or_bench_hard_1k}` run a second generation on the base route, capped at `GRADER_MAX_TOKENS` = 4096,
recorded per task as `grader_max_tokens` and hashed into the store identity.

## 8. Launcher — `scripts/modal/tinycap_served.py`

The wave names its Modal app (`--app-name`, also `TINYCAP_APP_NAME` in the environment of the `modal
run`; `contracts.app_name_for(purpose, model_slug)`); `contracts.MODAL_APP_NAME` = `tinycap-served` is the
sandbox orchestrator's app. Containers (`ALL_CONTAINER_NAMES`): `tinycap-e4b-prompted_safety-think_off`,
`tinycap-e4b-prompted_safety-think_on`, `tinycap-e4b-psrkl_safety-think_off`,
`tinycap-e4b-psrkl_safety-think_on`, `tinycap-sandbox-execute`; at most
`MAX_GPU_CONTAINERS` GPU containers without `--allow-extra`. `--cells` takes any token
`contracts.parse_launcher_cell` accepts (`none`, `prompted__<si>`, `<si>__<arm>`, the Class B ids),
translated with the one TrainTest → harness vocabulary and validated by `defense_cells.parse_cell_id`.
`--modes` shares `LAUNCHER_THINKING_MODES` with the driver. The launcher composes the driver argv with
`contracts.driver_argv` (the `TINYCAP_DRIVER_FLAGS` plus the only-when-given flags: the override flags
of §11, `--on-cap-by-task`, `--item-subset`, `--attempt-id`, `--row-engine-record`) and the standup argv
with `contracts.standup_extra_argv`; the preflight's argparse probe (`DRIVER_PROBE_VALUES`) checks every
flag against the staged driver. Expected identity (`ExpectedIdentity`, `expected_identity(...)`) is
derived launcher-side and `untrusted_store_reason` compares it field by field with a store's manifest.
The staged tree's `TREE_SHA` stamp is written at upload; `writer_tree_sha_of(current_tree_src())` is the
writing tree's commit in the merge REC.

## 9. Sandbox — `scripts/modal/tinycap_sandbox.py`

`modal.Sandbox` per (cell, mode, dataset) on
`ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740`
(`SANDBOX_IMAGE_DIGEST`); `SANDBOX_CREATE_KWARGS_REQUIRED` = `block_network=True, cpu=8, memory=16384,
timeout=3600`. Samples at `/work/samples.jsonl`, results at `/work/samples_eval_results.json`, validated
(`base` / `plus` per task id, exact id set) before anything touches the volume. The done marker is
authoritative for pending / resume; a validated results file makes a job stage `convert`, an invalid one
stage `execute`. The generation writer records the sha256 of the exact `samples.jsonl` it wrote
(`samples_sha256`, lifted into the DONE marker). The re-bind mode (`--rebind-samples-digest` /
`plan_rebind` / `run_rebind`) replaces a DONE entry's samples digest when every witness
(`store_summary`, `done_marker`, …) agrees, stamps `rebound_at` on the marker, and the merge records it
under `rebound` (`MERGE_REBIND_RECORD_FIELDS`); the canonical JSON every witness is compared through is
`completion.canonical_json_text` (`atomic_write_json`'s exact `json.dumps` call).

## 10. Client lock

`configs/capeval/tinycap-client-requirements.in` (`CLIENT_LOCK_IN`) → `.txt` via `uv pip compile
--no-config --generate-hashes`; installed `--no-deps --require-hashes` into `/opt/client`. Top-level
pins (`CLIENT_TOP_LEVEL_PINS`): inspect_ai 0.3.260, inspect_evals 0.18.0, evalplus 0.3.1,
datasets 5.0.1, aiohttp 3.14.3, transformers 5.15.0, inspect-sandboxes 0.5.0, swebench 5.0.2,
immutabledict 4.3.1, torch 2.13.0 and scipy 1.18.1 (the driver imports both; CPU use only); the
IFEval checker rides as the one VCS line (`CLIENT_VCS_PINS`). The subset the driver enforces at
startup is `FRAMEWORK_PINS`.

## 11. The out-of-support-thinking override — `--allow-out-of-support-thinking`

Every fitted object was fit under thinking-OFF, one-turn renders; serving one under a thinking-ON
render is out of support. `FLAG_ALLOW_OOS_THINKING` = `--allow-out-of-support-thinking` authorises
exactly one experiment, `AUTHORISED_OOS_THINKING` (`AuthorisedOutOfSupportExperiment`, fields
`OOS_AUTHORISED_FIELDS` = `model, cell, controls, mode, sampling_mode, samples, reduction_mode, pass_no,
max_tokens_on`): qwen35_4b × {`none`, `safety_refusal__psr_kl_all`} × {off, on}, paired-200 GSM8K, k = 1
at the card row, `per_row`, pass 1, `max_tokens_on {gsm8k_paired200: 8192, gsm8k: 8192}`
(`OOS_THINKING_MAX_TOKENS` = 8192), `prompt_opens_think_on = True`. `check_oos_authorised` runs at the
launcher's plan, in the worker and at the driver's startup; the run-ending refusals are `OOS_REC_NAMES` =
`OOS_UNAUTHORISED, OOS_DECLARATION_UNATTESTED, OOS_DECLARATION_MISMATCH, OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH,
OOS_SALT_POLICY_UNEXPECTED, OOS_TORN_STORE_UNREPAIRABLE, THINKING_UNSERVABLE, REDUCTION_MODE_UNATTESTED,
REDUCTION_MODE_MISMATCH` (`REC_OOS_*`, `REC_THINKING_UNSERVABLE`, `REC_REDUCTION_MODE_*`; defined once
in `contracts.py`, re-exported by reference in the launcher; `OutOfSupportAttestationRefusal(rec, msg)`).

**What the lens steers under the override.** On Qwen3.5 the assistant header moves with the render:

| regime | rendered header | H | ids |
|---|---|---|---|
| OFF (the fit regime) | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | 7 | `[248045, 74455, 198, 248068, 271, 248069, 271]` |
| ON (served under the override) | `<\|im_start\|>assistant\n<think>\n` | 5 | `[248045, 74455, 198, 248068, 198]` |

(`QWEN35_THINK_OPEN_ID` = 248068, `QWEN35_THINK_CLOSE_ID` = 248069.) A header-located object served
ON is therefore declared `serve_out_of_support: ["thinking"]` by name — the cell is spelled
`<cell>@thinking` — and the served header is certified and attested back by every worker
(`docs/served_thinking_on.md`). On a native (LoRA-engine) table the zero cell is `none__lora0`
(`TWIN_CELL_ID_LORA`; `twin_cell_id_for(engine)` picks the twin for the served engine). The stamp triple every row carries (`STAMP_BLOCK_KEY` = `petri`;
`EXPECTED_STAMP_TRIPLE_FIELDS` = `salt_regime, resumed_policy, render_regime`; `stamp_regimes(row)`
refuses a present malformed value and reads only a missing key as absent; `STAMP_VOCABULARY`).

**The reconciled names** — the contract's placeholders as `contracts.py` names (the vocabulary literals
`@`, `thinking`, `serve_out_of_support`, `serve_experiment` are imported from
`sisteer.capeval.serving.serve_declaration`, never retyped). The value column is pinned to the
constants by `tests/test_tinycap_contracts.py`:

| contract placeholder | `contracts.py` name | value |
|---|---|---|
| `<DECL_KEY>` / `<DECL_VALUE>` | `OOS_THINKING_DECLARATION_KEY` / `OOS_THINKING_DECLARATION_VALUE` | `petri_thinking_on` / `True` (a `bool`) |
| `<STANDUP_CELL_SPELLING>` | `OOS_STANDUP_CELL_SUFFIX` (= `VARIANT_SEP + VARIANT_THINKING`) | `@thinking`, no new flag |
| `<EXPECT_SHA_SPELLING>` | `OOS_EXPECT_SHA_KEY` | `traintest_variant` (the keyed form; `bare` legal, never emitted) |
| `<ATTEST_KEY>` / `<ATTEST_ERROR_KEY>` | `PROXY_IDENTITY_SERVE_DECLARATIONS` / `…_ERROR` | `lens_serve_declarations` / `lens_serve_declarations_error` |
| `<ATTEST_MAP_KEY>` | `cell_spellings(cell, declared=True).harness`; `TWIN_CELL_ID` | `si_safety__psr_kl_all@thinking`; `none__lens0` (the engine's zero cell — `none__lora0` / `TWIN_CELL_ID_LORA` on a native table, `twin_cell_id_for`) |
| `<DECL_EXPECTED>` | `OOS_CELL_DECLARATION_EXPECTED` | `["thinking"]` |
| `<ROUTE_KEY>` | `ROUTE_KEY_SERVE_DECLARATIONS` | `lens_serve_declarations` (the whole map, no derived scalar) |
| `<THINKING_HEADER_SOURCE>` (+ its error) | `THINKING_HEADER_SOURCE` / `THINKING_HEADER_ERROR_SOURCE` | `/v1/models.petri_proxy.thinking_header` / `.thinking_header_error` |
| `<THINKING_HEADER_SWITCH>` | `THINKING_HEADER_SWITCH_FIELD`, `…_SWITCH_EXPECTED`, `…_MOVES_EXPECTED` | `switch`, `True`, `True` (JSON booleans, compared as typed) |
| `<STAMP_RENDER_REGIME>` | `STAMP_KEY_RENDER_REGIME`, `RENDER_REGIME_LITERALS`, `RENDER_REGIME_BY_ARM` | `render_regime` ∈ {`thinking_on`, `thinking_off`, `unspecified`} |
| `<STAMP_SALT_REGIME>` | `STAMP_KEY_SALT_REGIME`, `STAMP_KEY_RESUMED_POLICY`, `SALT_REGIME_LITERALS`, `DECLARED_SALT_POLICY_LABEL`, `FORCED_SALT_POLICY_LABEL`, `RESUMED_POLICY_LABELS` | `cache_salt_regime` ∈ {`cell-stable`, `request-unique`}; `resumed_prefill_policy`'s eight labels incl. `declared-request-unique` and `forced-request-unique` |
| the refusal matrix | `THINKING_REFUSAL_CODES` | the 9 codes of the proxy's thinking matrix (`docs/served_thinking_on.md`) |
| the endpoint's additive keys | `ENDPOINT_KEY_CHAT_TEMPLATE_HASH`, `ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS`, `ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256(_BY_CELL)`, `ENDPOINT_KEY_LENS_SALT_POLICY`, `LENS_SALT_POLICY_ATTESTED`, `OOS_EXPECTED_RESUMED_PREFILL_POLICY` | `chat_template_hash`, `resumed_prefill_policy_cells`, `object_artifact_sha256(_by_cell)`, `lens_salt_policy`; `attested`; `conditional` |
| the artifact on the driver | `FLAG_OOS_PROMPT_BUDGET` (one constant: the launcher's dry run, its argparse probe and the driver's parser all read it) | `--oos-prompt-budget <out-root>/oos_prompt_budget.json` |
| the artifact digest on the driver | `FLAG_EXPECT_OBJECT_SHA256` | `--expect-object-sha256 <64-hex>` — on the out-of-support LENS configuration only, the same digest the launcher held the standup to; the driver refuses such a run told none (`OOS_DECLARATION_UNATTESTED`) |
| the run-ending RECs | `OOS_REC_NAMES` (`REC_OOS_*`, `REC_THINKING_UNSERVABLE`, `REC_REDUCTION_MODE_*`) | defined once in `contracts.py`; the launcher's names are re-exports by reference |
| the comparator's flag (the SECOND override) | `FLAG_ALLOW_OFF_CAP_OVERRIDE`; `OFF_CAP_OVERRIDE_NOTE` | `--allow-off-cap-override` on the launcher AND the driver; NOT in `TINYCAP_DRIVER_FLAGS`; `driver_argv(allow_off_cap_override=…)` appends it in the OOS flag's slot |
| the comparator tuple | `AuthorisedOffCapComparator`, `AUTHORISED_OFF_CAP_COMPARATOR`, `OFFCAP_AUTHORISED_FIELDS`, `OFF_CAP_OVERRIDE_MAX_TOKENS`, `off_cap_override_max_tokens`, `check_off_cap_override_authorised`, `OffCapOverrideUnauthorised`, `REC_OFFCAP_UNAUTHORISED` | qwen35_4b × {`none`, `safety_refusal__psr_kl_all`} × {`off`} × GSM8K, k = 1, `per_row`, `max_tokens_off {gsm8k_paired200: 8192, gsm8k: 8192}`; `mode` must be `off`; `OFFCAP_UNAUTHORISED` is a plan-time refusal, not one of `OOS_REC_NAMES` |
| the flags' exclusivity | `OverrideFlagsExclusive`, `check_override_flags_exclusive` | both override flags in one invocation refuse naming both — at the launcher's `main`, `plan_configurations`, the driver's startup and `tinycap_identity_fields` |
| the comparator's identity shape | `TINYCAP_IDENTITY_OFF_CAP_FIELDS`; `tinycap_identity_fields(off_cap_override=True)` | `(max_tokens_by_task,)` alone — a present hashed cap IS the positive producer marker; `completion.store_has_cap_override` / `completion.store_requires_outcomes` read it (`held_to_record = oos_run or offcap_run`) |
| the OFF artifact at the override cap | `OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE`; `prompt_budget_matches_run` | `scripts/tinycap_oos_prompt_budget.py --arm off --off-cap-override` writes `arm off, cap 8192`; the artifact's arm and cap are held to the run's at the preflight, in the worker and in the driver; one artifact path per pin (`_preflight/<model>/<pin40>/`) |
| the base-route negative probe | `OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED` | `evidence.base_route_declaration_refused = {vmodel, status, code, message, ok, expected}` on the comparator's `none` driver only: one declared ON body to the base vmodel must be refused 400 `thinking_on_not_lens` before any forward |

**The thinking-OFF comparator at the thinking-ON cap — `--allow-off-cap-override`.** The second
override re-runs the experiment's two thinking-OFF legs at the thinking-ON cap, so each cross-arm
shift of the 2×2 decomposes into a budget term (OFF/8192 − OFF/512, same render) and a render term
(ON/8192 − OFF/8192, same cap). It changes exactly two things for a configuration it authorises: the
executed thinking-OFF cap of the named tasks (`inspect_served.task_max_tokens(off_cap_override=True)` →
`(8192, "offcap_override")`; `cap_source offcap_override`) and the hashed identity key
`max_tokens_by_task`. It changes nothing about the render: no declaration on any route,
`enable_thinking: false` explicit on every request, `render_regime thinking_off`. Its stores land under
their own out-suffix (`offcap-<pin7>`) and a different `harness_config_hash`, so neither override's
stores resume onto the other's.

## 12. MATH-500 at one draw, thinking ON, cap 16,384

The four full22 models served thinking-ON on MATH-500 at one 16,384 draw, bare / prompted / fitted.
The cap is `MATH500_MAX_TOKENS_DEFAULT` (§1), the windows `capability:gen_long_math` and
`capability:gen_long_math_longsi` (§5), and MATH identities move with the cap by design
(`math500_max_tokens`); every non-MATH identity and geometry is byte-identical. Request evidence
version 2 (`request_evidence.jsonl`) is wire-bound and reconciled with the ledger by attempt.

## 13. GSM8K paired-200 on the paper's objects — the bounded on-cap setter and the rung store

| name | meaning |
|---|---|
| `FLAG_ON_CAP_BY_TASK` = `--on-cap-by-task <task>=<cap>` | the one mechanism that raises a thinking-ON task's cap; `MANIFEST_KEY_ON_CAP_BY_TASK` = `on_cap_by_task` records it; `CAP_SOURCE_ON_CAP_EXPLICIT` = `on_cap_explicit` |
| `ON_CAP_BY_TASK_ALLOWED` = `{gsm8k_paired200: (8192, 16384), math: (16384,)}`, `ON_CAP_CEILING` = 16,384 | the admissible caps per task; `check_on_cap_setter` refuses anything else (`OnCapSetterRefused`) |
| `G8K_EXPERIMENT_NAMES`, `g8k_tuple_name(model, cap, declared=)` | the authorised experiment tuples (`gsm8k_paired200_<model>_<cap>`, `_declared_`, `_steer_` variants) added to `AUTHORISED_OOS_EXPERIMENTS` |
| `G8K_ALL_POSITION_CELLS` | the six all-position cells: `constv2-mix-full__psr_ce_all`, `constv2-mix-full__lora_r4_ce`, `constv2ctl2-full__psr_ce_all`, `constv2ctl2-full__lora_r4_ce`, `constv2-mix-full__psr_ce_all_minus_ctl_distilled`, `constv2-mix-full__lora_r4_ce_minus_ctl` |
| `G8K_HEADER_LOCATED_CELLS` | `safety_refusal__psr_kl_all`, `safety_refusal__lora_r4` — served on the declared path on the header-moving families |
| `header_invariant_declaration_refusal(family, header_located=, arm=)`, `header_invariant_declared_cells(model_key)` | the first is retained with its signature and never refuses (a header-located cell on a header-moving family serves declared); the second lists the registry's declared cells for such a family |
| `G8K_STAMPS_BASE` = `(cell-stable, not-consulted, thinking_on)`, `G8K_STAMPS_FITTED` = `(cell-stable, always, thinking_on)`, `G8K_STAMPS_DECLARED` = `(request-unique, declared-request-unique, thinking_on)`; `G8K_STAMPS_BY_CLASS`, `g8k_expected_stamps(fitted=, declared=)` | the stamp triple per cell class, derived, never spelled by a reader |
| `THINKING_HEADER_MOVES_EXPECTED` = `True`, `header_moves` | the proxy's header fact a declared run requires |
| `FLAG_ITEM_SUBSET` = `--item-subset <path>`, `ITEM_SUBSET_SCHEMA` = `rung_subset/1`, `ITEM_SUBSET_FIELDS` = `schema, task, source_out_root, source_identity_hash, source_cap, cap, rule, threshold, ids, n_at_cap, n_non_terminating, n_boundary, transport_check` | the rung's item subset file (`read_item_subset`, `validate_item_subset`, `ItemSubsetError`); `item_subset_sha16` = sha256 of the sorted unique ids' JSON, 16 hex |
| `IDENTITY_KEY_CAP_RUNG` = `cap_rung`, `CAP_RUNG_FIELDS` = `source_identity_hash, subset_sha16, source_cap, rule`, `cap_rung_identity(doc)` | the rung store's hashed identity, so a rung store never resumes onto its source; `MANIFEST_KEY_ITEM_SUBSET` = `item_subset` records the block (`item_subset_block`) |
| `inspect_rung` (`sisteer.capeval.tiers.inspect_rung`), `classify_store(out_root, *, task=, …)` | reads a source store and classifies its draws: `ITEM_KINDS` = `truncated, non_terminating`; `ITEM_FACT_FIELDS` the keys of one `item_facts` record; `RECORD_FIELD_ELIGIBLE_IDS` = `eligible_ids` |
| `GSM8K_TRIGGER_RULE`, `GSM8K_TRIGGER_THRESHOLD`, `GSM8K_REPORT_LINE` | `trigger iff n_at_cap / n > threshold`; the eligible set = at-cap − degenerate − boundary; ceiling 16,384; `GSM8K_REPORT_LINE` is a report field, never a gate |
| `NOT_TRIGGERED_EXITS` = `clean, below_threshold`; `TRIGGERED_TERMINAL_EXITS` = `triggered_no_eligible, transport_refused`; `EXIT_ESCALATE` | the rung classifier's exits; `transport_check` = `cap_ladder.transport_check(...).as_dict()` must be `ok: true` for a subset to be a rung |
| `FLAG_ATTEMPT_ID` = `--attempt-id a<k>`, `ATTEMPT_ID_FIELD` = `attempt_id`, `ATTEMPT_KINDS` = `generation, finalisation`, `AttemptScope` | the attempt-scoped producer contract: the wrapper mints one id per attempt and the ledger stamps it on every row |
| `RESUME_ATTEMPT_SCOPED_FIELDS` = `attempt_id, kind, generation_calls, finalised_from_attempt, started, ended, terminal` | one entry per attempt in the store's resume record |
| `MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT` = `accepted_generation_attempt`, `MANIFEST_KEY_ATTEMPT_ATTRIBUTION` = `attempt_attribution`, `ATTEMPT_ATTRIBUTION_FIELDS` = `n_rows, rejected, ledger_blocked, refusals, task_calls, probe_calls, kind, terminal` | which attempt's generations the store accepted, and the per-attempt attribution of ledger rows (`ledger_row_attempt`, `resolve_attempt_kinds`, `attempt_attribution`, `scoped_rejected_count`) |

## 14. BigCodeBench and BFCL — the full legs, the owned sandbox and the `protocol` sampling mode

`bigcodebench_full` — every `v0.1.2` complete-split id, 1,140, Hard included, pinned by id in
`configs/capeval/tasks/bcb_full.json` (`BCB_FULL_PIN_PATH`, `load_bcb_full_pin`, `BCB_FULL_N`) — runs in
OWNED Modal sandboxes (`tiers/sandbox_owned.py`, `modal_owned`) with a watcher
(`scripts/tinycap_sandbox_watch.py`) and a proxy timeout (`proxy_upstream_timeout_s`); `bfcl_full` — every
item of the battery's 18 single-turn categories, 3,981, pinned in `configs/capeval/tasks/bfcl_full.json`
(`BFCL_FULL_PIN_PATH`, `load_bfcl_full_pin`, `BFCL_FULL_N`) — needs a `--tool-calls` standup. The two are the
paper's item sets (`FULL_ROW_TASKS`, identity keys `IDENTITY_KEY_BCB_FULL_ITEM_SET` / `IDENTITY_KEY_BFCL_FULL_ITEM_SET`,
filed by `files_bigcodebench_full` / `files_bfcl_full`) and the front door's (`reproduce/07_bcb_bfcl.sh`). A
run filing a full leg carries the harness limits of `FULL_TASK_LIMITS`, picked by the configuration — the
driver bound `FULL_DRIVER_TIMEOUT_S` (8 h, `driver_timeout_s_for`) and the GPU function's
`FULL_FUNCTION_TIMEOUT_S` (10 h); the sandbox, verifier and client timeouts and the per-container sandbox
count are unchanged — at the completion caps `BCB_FULL_MAX_TOKENS_OFF` / `BFCL_FULL_MAX_TOKENS_OFF` (4,096 and
1,024: the instrument's, not the item set's). BigCodeBench rows carry `code_source`, `sandbox_status`,
`verifier_explanation` (`BCB_ROW_FACT_FIELDS`); its generation and sandbox verification may run as two phases
(§16, `SANDBOXED_ROW_TASKS`).

**The `protocol` sampling mode — each leg under its own reference decoding** (`SAMPLING_PROTOCOL` =
`protocol`): `contracts.PROTOCOL_SAMPLING_BY_TASK: {row task: ProtocolRow}` where `ProtocolRow(executed:
SamplingRow, protocol_temperature: float, executed_reason: str)` keeps apart what the benchmark declares
(`protocol_temperature`, recorded) and what this stack executes (`executed`, hashed): BigCodeBench greedy
(0.0), BFCL `BFCL_PROTOCOL_TEMPERATURE` = 0.001, `PROTOCOL_TOP_P` 1.0, `PROTOCOL_TOP_K` 0. vLLM pins a
minimum temperature `VLLM_MAX_TEMP_PINNED` = 0.01, so BFCL's 0.001 needs the clamp override
`SAMPLING_CLAMP_OVERRIDE` = `{from: 0.01, to: 0.001}`: `capctrl_serving_up.py --sampling-clamp-override 0.001`
(`FLAG_SAMPLING_CLAMP_OVERRIDE`), recorded in `endpoint.json` as `sampling_clamp_override`
(`ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE`) and in the hashed `served` block
(`contracts.served_identity_fields(protocol_sampling=)` = `SERVED_IDENTITY_FIELDS` +
`SERVED_IDENTITY_PROTOCOL_FIELDS`), applied through the leaf wrapper `sisteer.capeval.serving.vllm_entry`
(`VLLM_ENTRY_MODULE`; `--sisteer-max-temp` = `FLAG_VLLM_ENTRY_MAX_TEMP`; `_MAX_TEMP` is the patched
constant), whose `executed_reason` is `PROTOCOL_EXECUTED_REASON_CLAMP_OVERRIDE` =
`equal (clamp override applied)`. `vllm_entry` asserts the process topology the patch covers: `--api-server-count` above 1 and
`--numa-bind` are refused (`vllm_entry.spawn_topology_reason`), and `vllm_entry.guard_mp_context` asserts
the effective start method at process creation.

Entry points: `sampling_for_task(task, mode, model_key)` (the executed row under protocol, `sampling_for`
otherwise; `sampling_for(model_key, "protocol")` refuses), `protocol_rows_by_task(tasks)`,
`protocol_executed_by_task(tasks)`, `protocol_sampling_identity(executed_by_task, samples_by_task)`,
`sampling_identity_for_mode(mode, model_key, tasks, samples_by_task)`. Identity: `sampling_by_task`
(`IDENTITY_KEY_SAMPLING_BY_TASK`; `TINYCAP_IDENTITY_PROTOCOL_FIELDS`) — the executed rows over the filed
tasks. Manifest: `sampling.by_task[task]` (`SAMPLING_BLOCK_KEY_BY_TASK`) carries exactly
`PROTOCOL_BY_TASK_FIELDS` = `executed, protocol_temperature, executed_reason, effective`, built by
`protocol_manifest_by_task(tasks, effective_by_task)` with `protocol_source`; `effective` is the server's
resolved sampling (`PROTOCOL_EFFECTIVE_FIELDS` = `temperature, top_p, top_k, min_p, repetition_penalty,
presence_penalty, frequency_penalty`), re-asserted by `check_protocol_effective(record)`;
`ProtocolSamplingUnavailable` refuses a task with no protocol row at plan time, at the driver and in the
worker.

**`check_protocol_sampling`** is the in-image CPU check (`modal run scripts/modal/tinycap_served.py::check_protocol_sampling
--sha <sha> --model <slug> [--tasks …]`): it launches the leaf wrapper and verifies the executed
sampling; its record is `protocol_check.json` (`PROTOCOL_CHECK_FILENAME`; `--protocol-check` =
`FLAG_PROTOCOL_CHECK` on the driver; `driver_argv(sampling_mode=, protocol_check=)` spells `--sampling`
only when the mode is not the default). A failed live validation is durable:
`sampling_validation_failed.json` (`SAMPLING_VALIDATION_FAILED_FILENAME`; RECs
`SAMPLING_VALIDATION_FAILED`, `CLAMP_OVERRIDE_INEFFECTIVE`). The launch-time roster snapshot is immutable
(`ROSTER_SNAPSHOT_CONFLICT`; a retry appends to `lane_roster.retries.jsonl`, `SNAPSHOT_RETRY_FILENAME`);
a certified wave's membership is `--ladder-wave-cells` (`FLAG_LADDER_WAVE_CELLS`). The served launch
flags are recorded from the manifest's `serving_launch_flags` (`protocol_launch_flags_record`), read
through the authoritative reader `manifest.load_petri_spec`.

## 15. ifeval — `inspect/ifeval` on the served adapter

One opt-in atom `ifeval` producing `inspect/ifeval`: inspect_evals' stock task (`google/IFEval` @
`IFEVAL_DATASET_REVISION`, `IFEVAL_N` = 541; scorer `instruction_following`). The row's `acc` reads
`prompt_level_strict`; the five score values + `instruction_id_list` are `IFEVAL_ROW_FACT_FIELDS`.
`IFEVAL_SAMPLES_PER_ITEM` = 2 (`SEED_RULE_IFEVAL`), thinking off, caps `IFEVAL_CAP_LADDER` = 2,048 →
4,096 → 8,192 (`IFEVAL_MAX_TOKENS_OFF`, `IFEVAL_CAP_CEILING`), each cap its own geometry entry
(`IFEVAL_GEOMETRY_ENTRY_BY_CAP`; `IfevalGeometryPooled` is the pooled measurement record). Driver flags
(`IFEVAL_DRIVER_FLAGS`): `--sampling`, `--tinycap-identity-version`, `--max-tokens-by-task`,
`--off-cap-by-task`, `--item-subset`, `--bank-view`. The image bakes NLTK data
(`ifeval_nltk_bake_commands`, `NLTK_DATA` = `NLTK_DATA_DIR`). The dispatcher
(`scripts/tinycap_ifeval_dispatch.py`) reads `launcher.REC_OK_STATUSES` = `OK, DONE, PARTIAL_COMPLETE`
and `IFEVAL_DISPATCH_STATE` = `dispatch_state.json`.

## 16. The row engine — one standup per model row

One standup per model ROW serves every cell of the row on one engine (LoRA cells as adapters selected per
request, steering cells on the residual adapter slot, `none` as the base route), one driver per cell.

| name | meaning |
|---|---|
| `FLAG_ROW_ENGINE_RECORD` = `--row-engine-record <path>` | the row runner's per-cell record `srv/cells/<cell>/row_engine.json`, read by the driver through `contracts.read_row_engine_record` and copied into `manifest["row_engine"]` (`MANIFEST_KEY_ROW_ENGINE`), unhashed; appended last by `driver_argv(row_engine_record=…)` |
| `ROW_ENGINE_MANIFEST_FIELDS` = `row_engine, standup_id, engine_cells, max_loras, residual_slot_armed, compile_cache, row_dir, verification_pass`; `ROW_ENGINE_COMPILE_CACHE_FIELDS` = `hit, key, vllm_cache_dir` | the record's exact keys (`validate_row_engine_record`, `RowEngineRecordError`); absent, the manifest is byte-identical (golden `tests/fixtures/tinycap_row_engine/driver_manifest_native_gsm8k_no_flag.json`) |
| `FLAG_VERIFICATION_PHASE` = `--verification-phase coupled\|generate\|verify`, `PENDING_MARKER_FIELDS` = `task, cell, n_items, n_expected_full, generated_at, log, generation_fcid` | the two-phase BigCodeBench contract: generation on the GPU, verification on CPU containers; `SANDBOXED_ROW_TASKS` = `inspect/bigcodebench_200`, `inspect/bigcodebench_full` |
| `scripts/modal/tinycap_compile_cache.py`: `cache_key(...)`, `with_compile_cache_env(...)`, `compile_cache_restore(key)`, `compile_cache_evidence(log, hash_dir)`, `compile_cache_persist(key)` | the warm compile cache: at most 7 tarballs under `/vol/tinycap/cache/compile/` plus `index.json`, unpacked node-local; `hit` is read off vLLM's own log lines, `restored` is not `hit` |
| `resolve_staged_tree(sha, …)`, `stage_tree_step(sha)`, `unpack_tree_local`, `tarball_tree_sha` | tarball-only staging of the code tree, unpacked node-local under `TREE_LOCAL_ROOT` |
| `INODES_PER_CELL_STORE` = 40, `ROW_DIR_INODES_BASE` = 12, `ROW_DIR_INODES_PER_CELL` = 7, `INODES_PER_ADAPTER` = 5, `COMPILE_CACHE_INODES_MAX` = 8; `forecast_inodes(n_cells, *, staged, n_new_adapters=0, n_rows=1)`, `probe_volume_inodes(mount)`, `inode_budget_verdict(used, total, forecast, floor=0.90)` | the volume's inode budget: a launch is refused (`INODE_BUDGET_EXCEEDED`, preflight check `inode_budget`) when `(used + forecast) / total >= floor` |
| `close_wave(inventory_json, confirm=False)`, `CLOSE_WAVE_INVENTORY_SCHEMA` = `tinycap_close_wave_inventory/1`, `CLOSE_WAVE_KINDS` = `tarball, parked_store, negctl_store, preflight_artefact, row_dir` | the wave-close prune over a volume-relative inventory; the planner refuses unless every launch-set row has a PASS gate record and every store is mirrored |
| `<gen root>/rows/<row>/<primary suffix>/` (`ROWS_SUBDIR`) | the row dir, outside `native/`, mirrored as a record, never as a store; the row pointer `row_dir/watch/attempt_<fcid>.json` and per-cell certificates `sweep_<fcid>__<cell>.json` are the watcher's row mode |
