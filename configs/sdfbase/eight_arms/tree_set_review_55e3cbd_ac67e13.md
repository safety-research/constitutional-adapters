# Tree-set review — MATH-500 thinking-ON: the published tree 55e3cbdd beside the extension wave tree ac67e138

Reviewed 2026-09-13 for the eight-arm extension (plan P5 / §4.5 / R21; `tinycap_math500_thinking.py --tree-diff-allow-file`).
The published 24 legs (rows 0–23) were produced at `55e3cbdd0bfb5d53f8d62c50a099a3c1c105100b`; the nine extension legs (rows 24–32:
per gemma4e4b / qwen35_4b / nemotron3nano4b the `prompted__si_constitution`, `constv2ctl2-full__lora_r4_ce` and
`constv2-mix-full__lora_r4_ce` cells, `thinking_card`) at `ac67e138b90aaefb3d7b8b9adeb5ae520404158a`. Between the two trees main
moved by 709 commits (553 non-merge), and `git diff --no-renames --name-only` names 982 paths. This file
ADMITS that diff; it is evidence, not the gate (R21). The gates are (a) the P11 compatibility preflight, run from the checkout whose
HEAD is ac67e138 against the published `none` legs' manifests and PASSED for all three panel models with no skipped or undetermined
check (`compat_<key>.json`: `head_equals_wave_sha true, dirty_paths [], skipped_checks [], problems [], undetermined []`), and
(b) the analysis's own per-contrast compatibility check over the produced identities (the shared projection, each row's arm contract,
the normalised server settings read off every leg's launch attestation, producer membership), which refuses by name on any difference.

What P11 compared and found EQUAL between a published `none` manifest and the wave tree's composed identity, per panel model (the
`shared` block): model id + revision (resolved and requested), `chat_template_hash`, `client_lock_sha256`, `identity_version` (2),
`inspect_ai` 0.3.260 / `inspect_evals` 0.18.0, the 500 item ids, `math500_max_tokens` 16,384 with `retry_max_tokens` 0,
`max_tokens_by_task`, `samples_by_task`, `sampling_mode thinking_card`, the numeric sampling projection (`SAMPLING_PARAM_KEYS`), the
`enable_thinking` request flag; the `server_settings` block: the served model / revision, `reasoning_parser`, `reduction_mode stacked`,
`server_default_thinking off` at both trees, the thinking header ids; the `sampler_resolution` block: every key a MATH-500 request omits
resolves to the same value at both trees — for gemma and nemotron vLLM's neutral default with no generation-config override on either
side; for qwen the ONE omitted key `repetition_penalty` is 1.0 under the published tree's neutral defaults AND under the wave tree's
composed `--override-generation-config` (the vendor instruct row pinned as the server default because the snapshot ships no
`generation_config.json`, main e631261d — the literal fork, admitted by effect; review H2: any key of that override the effect
rule does not understand refuses the contrast). Reviewed again 2026-09-13 after the record's opening review (revision 2: the three
lines below marked "revision 2" corrected stale or wrong evidence; nothing else in this file moved). The two intentional per-row differences (the constitution
rows' larger window `capability:gen_long_math_longsi` L 24,064, and `--enable-log-requests`) are validated against each row's own
contract, not compared. The vLLM serving requirements (`configs/capeval/vllm-requirements.txt`, `-nemotron.txt`) and the client lock
(`tinycap-client-requirements.txt`) are byte-identical at the two trees (absent from the diff), so both serving venvs are the same.

The MATH-500 launch path at the wave tree, read from the sbatch and the shell it sources: `math500_thinking.sbatch` → `m5t_common.sh`
→ `full22/common.sh` (+ `load_check.py`, `smoke_positive_control.py`, `positive_control_verdict.py`, `prompted_window_check.py`,
`served_thinking.sh`, `roster_view.py`, `roster_cell_count.py`, `make_bank_view.py`) → `mt_serving_up.py` (+ `capctrl_serving_up.py`)
→ `served_capability_run.py`; `cells.py`, `make_math500_overlays.py`, `nemo_backend_probe.py`, `record_resolved_sampler.py`,
`tinycap_oos_prompt_budget.py` beside it; the Python modules `sisteer.capeval.serving.*`, `sisteer.capeval.tinycap.contracts`,
`sisteer.capeval.tiers.inspect_served`, `sisteer.capeval.petri.registry_adapter`, `sisteer.chat_markers`, `sisteer.models`.

Every glob below carries its one-line evidence; the grammar is the loader's (a bullet whose first backticked token is the glob,
then a dash, then the evidence). Producer paths first, then everything else. A path may match more than one glob; the specific
line is the one that applies.

## Producer paths (the served identity and the store)

- `scripts/served_capability_run.py` — the driver (30 commits): the per-kind probe TAG header on its own probes (logging only), the sampled-controls extraction into `sisteer.eval.sampled_controls` (the judged over-refusal controls, not MATH-500), the served==trained template assertion moved BEFORE the first steered request (a preflight that refuses, never changes a request), the ladder / rung machinery and `item_subset` manifest keys (GSM8K's; MATH-500 runs one pass at 16,384), retired decode-based MMLU / HellaSwag / full BFCL / AlpacaEval / WMDP tasks, the gemma-4 admission of an ON header EQUAL to the OFF header (the published gemma legs already served that equality: P11 `thinking_header_ids` on == off at both trees); every shared field it composes is in P11's `shared` block and EQUAL.
- `src/sisteer/capeval/serving/vllm_server.py` — the vLLM launcher (21 commits): the launch flags it composes are exactly what every leg's `vllm_server.log` startup echo carries, and the analysis's `launch_attestation` compares every echoed flag outside the named geometry / logging exclusions between the published `none` and each extension leg (`server_settings_equal`); P11's `server_settings.compared` block is EQUAL and its `composed_only` block records the wave's dtype bfloat16, no quantisation, `default_chat_template_kwargs {"enable_thinking": false}`, gpu_memory_utilization 0.9, and — revision 2 (2026-09-13), corrected from "prefix caching on, no generation-config override" — prefix caching per MODEL as the composer pins it (`enable_prefix_caching` True on the gemma and GLM legs, False on every qwen and nemotron leg, read from all 33 startup echoes; equal per model between the trees) and, on the three extension qwen standups only, the composed `--override-generation-config {min_p 0.0, repetition_penalty 1.0, temperature 0.7, top_k 20, top_p 0.8}` (the vendor instruct row, main e631261d; the published qwen standups carry none) — the literal fork the effect rule admits, and which the analysis reports as `server_settings_literal_equal false`; gemma's and nemotron's standups carry no override on either tree.
- `src/sisteer/capeval/serving/proxy.py` — the proxy (12 commits: hop timeouts derived from the ladder instead of the 600-s transport default, attestation plumbing, request logging): the transport, never a request body or sampler; the client timeout is recorded per leg (`summary.client_timeout_s`) and the ledger equality holds on every extension leg.
- `src/sisteer/capeval/serving/lens_worker.py` — the lens route the six LoRA legs are served through: the served object sha is asserted against the row's `--expect-object-sha256` at standup and compared to the (model, arm) pin by the analysis's identity + certification checks; the six pins are byte-equal at both trees (`configs/sdfbase_full22_pins.json`, below).
- `src/sisteer/capeval/serving/lora_export.py` — the LoRA export / native multi-LoRA path (7 commits): the six LoRA legs' `served_positions` and `reduction_mode stacked` are the row's expectation and equal to the published legs' (P11 `server_settings.reduction_mode` EQUAL); the object bytes are pinned by sha.
- `src/sisteer/capeval/serving/serving_attestation.py` — NEW: writes `steering_attestation.json` beside every standup (job id, port, argv sha); a recorded field the analysis reads to bind a log to a leg, never a served value.
- `src/sisteer/capeval/serving/mamba_backend.py` — NEW: the nemotron fused-Mamba backend probe / attestation; the nemotron extension legs ran on the attested backend the published nemotron legs already used (the wave root's `nemo_backend/mamba_backend_attestation.json` reads admitted, fused, `serving_requirements_sha256` equal to the tree's vLLM requirements hash; the published nemotron logs show the same kernel overlay 2.3.2.post1 / 1.6.2.post1).
- `src/sisteer/capeval/serving/vllm_entry.py` — NEW entry module for the engine process (5 commits): how the process is started, not what it serves; the served identity is the echoed argv compared above.
- `src/sisteer/capeval/serving/vllm_shim/sitecustomize.py` — NEW: the engine-process sampling-CLAMP override for protocol-mode launches — revision 2 (2026-09-13), corrected from "records the resolved sampler per request … logging only", which it never was: armed ONLY by `SISTEER_VLLM_MAX_TEMP`, it lowers vLLM 0.25.1's `_MAX_TEMP` clamp in every spawned interpreter so a BFCL request at temperature 0.001 decodes literally (`vllm_server.apply_sampling_clamp_override`, passed a value only for protocol-sampling launches and `None` for every other launch); no MATH-500 standup arms it (none of the 33 server logs carries the arm or the shim's signature), and MATH-500's rows decode at temperature 0.6–1.0, above any clamp. The per-request sampler readback is vLLM's own `--enable-log-requests` line, read by `record_resolved_sampler.py`, not this file.
- `src/sisteer/capeval/serving/standup_thinking.py` — the thinking-ON header certification (1 commit): certifies the header at standup and refuses otherwise; the header ids are EQUAL at both trees (P11 `thinking_header_ids`).
- `src/sisteer/capeval/serving/run_identity.py` — the run identity block (1 commit): `identity_version` 2 at both trees (P11 `shared.identity_version`).
- `src/sisteer/capeval/serving/usage_ledger.py` — the usage ledger writer (3 commits; the `thinking` and `request_id` columns): recorded fields; the ledger equality (cap-hit == at-cap) holds on all nine extension legs.
- `src/sisteer/capeval/serving/geometry.py` — the served-geometry registry (7 commits): adds `capability:gen_long_math_longsi` (L 24,064) for the constitution rows and other lanes' entries; `capability:gen_long_math` (17,664) is unchanged and the geometry is validated per row, not compared (P5 (i′)); the budget fit sidecars at ac67e138 read `fits: true` on all six extension classes. Revision 2 (2026-09-13) — the qwen35_4b BUDGET moved between the trees, and a grep of the main table does not show it: commit 31b869f7 (2026-09-06) gave qwen35_4b a PROVISIONAL ceiling of exactly half its swept budget (360,448 vs 720,896, `source=PROVISIONAL_BUDGET_SOURCE`) in the separate `_PROVISIONAL_BUDGETS` structure that SHADOWS the main table's row (which reads 720,896 at every revision), the wave tree inherited that structure through 2beb802d (2026-09-10), and 8ec7288e (2026-09-11) lifted the shadow after six lens-stack standups exercised the full 720,896. 31b869f7 is an ancestor of the published tree 55e3cbdd (whose `geometry.py` carries `batched_token_budget=360_448` for qwen35_4b) and 8ec7288e of the wave tree ac67e138 (no shadow), so the published qwen legs were served at S 20 (20 × 17,664 = 353,280 ≤ 360,448) and the extension qwen legs at S 40 (706,560 ≤ 720,896) at the SAME window, the constitution row at L 24,064 / S 29 (697,856); gemma and nemotron kept their budgets (S 33 / 44 at 17,664; the constitution rows S 24 / 32 at 24,064). RULE: served concurrency is deliberately NOT an identity axis for a VALUE — a re-randomised sample of the same estimand at the same window is noise, not a second instrument — so the qwen contrasts are admissible comparisons of values and are disclosed as a single-run cross-geometry comparison in the results record (section 15) and on every figure and table cell they reach; it IS decisive for any BIT-IDENTITY or byte-reproduction claim (batch composition sets the bits under bf16 non-invariance), so this file and the record make NO byte-level reproduction claim spanning the S 20 / S 40 boundary — the extension's only reproduction checks (the published report's leaf-for-leaf reproduction through the new code paths, `tests/test_eight_arms_frozen_reproduction.py`; run E against run F on one pull) each sit inside ONE tree, and the per-kind recorder's `AGREES` compares resolved sampler VALUES, never bytes. A matched re-run of the three qwen legs at S 20 (≈ 0.5 GPU-h) is an open option, not launched here.
- `src/sisteer/capeval/serving/**` — the remaining serving modules (`fidelity.py`, `lens_adapters.py`, `lens_steer.py`, `native_worker.py`, `proxy_engine.py`, `registry_shim.py`, `resolved_spec.py`, `serve_declaration.py`, `worker_reports.py`): worker reporting, fidelity gates and adapters on the lens route; none composes a launch flag or a request parameter outside what the attestation echo and the shared projection compare.
- `src/sisteer/capeval/tiers/inspect_served.py` — the Inspect adapter (10 commits): the 500 item ids, the cap 16,384 with retry 0, `samples_by_task {inspect/math: 1}`, `inspect_ai` 0.3.260 / `inspect_evals` 0.18.0 are EQUAL at both trees (P11 `shared`); the `on_cap_by_task` / `not_available` machinery is other lanes' (llama, GSM8K).
- `src/sisteer/capeval/tiers/**` — the remaining tier modules (`at_cap.py`, `cap_ladder.py`, `carry_over.py`, `inspect_rung.py`, `judged.py`, `sandbox_owned.py`, `served_tiny.py`): the truncation ladder and rung adapters (GSM8K / the sampled ladder; MATH-500 has no rung), the judged tiers and sandbox-owned legs (other instruments); not on the MATH-500 path.
- `src/sisteer/capeval/tinycap/contracts.py` — the tinycap contracts (17 commits): adds manifest keys (`on_cap_by_task`, `item_subset`, `accepted_generation_attempt`, `attempt_attribution`), the eight-arm tuples and the GSM8K / llama contracts; `SAMPLING_PARAM_KEYS`, `identity_version`, the `sampling_ref` construction and the MATH-500 tuple bodies are unchanged (P11 rebuilt the published `sampling_ref` through this tree's `contracts.sampling_ref` and it matched the manifest's; the three `_lora2` tuples equal `_m5t_tuple(key, thinking_card)` in every field but the cell list, pinned by test).
- `src/sisteer/capeval/tinycap/object_identity.py` — the object identity check (1 commit): the served object sha vs the pin; the six extension pins are equal at both trees.
- `src/sisteer/capeval/tinycap/**` — the remaining tinycap modules (`protocol_check.py`, `rung_probe.py`, `sandbox_scope.py`, `verifier_transport.py`, `widen_roster.py`, and the removed `terminal_bench.py`): the rung probe, the BigCodeBench / BFCL sandbox transport and the retired Terminal-Bench leg — other instruments.
- `src/sisteer/capeval/manifest.py` — the manifest writer (6 commits): new RECORDED fields (attestation paths, request evidence, attempt attribution); the analysis reads the extension manifests through this schema and P11 projected the published manifest onto the same keys.
- `src/sisteer/capeval/judging.py` — the judged tiers' judge client (not the MATH-500 Inspect scorer).
- `src/sisteer/capeval/petri/registry_adapter.py` — the SI registry adapter (`harness_cell_id`, `_traintest_si_text_bytes`): the prompted constitution cell's harness id `prompted__constitution` and the SHORT SI bytes whose sha256 (21e06f48…) is asserted per request (H8) and equals the registry (P11 `arm_contracts.si.sha256_registry == sha256_row`).
- `src/sisteer/capeval/petri/**` — the remaining Petri modules (`defense_cells.py`, `petri_stats.py`, `scenarios.py`): the Petri instrument, not MATH-500.
- `src/sisteer/capeval/misalign/**` — the misalignment-evals instrument (new package); not on the MATH-500 path.
- `scripts/sdfbase/math500/cells.py` — E1's matrix: rows 0–23 regenerate byte-identically (`tests/fixtures/math500_matrix_55e3cbdd.json`); rows 24–32 appended with the row identity block, the `_lora2` tuples and the per-kind recorder constants; the published legs' identity excludes `explicit_cells` (R14 / F2).
- `scripts/sdfbase/math500/m5t_common.sh` — E1's launcher shell: the row's geometry passed to the standup, `--enable-log-requests` (logging only) and the per-kind recorder argv (a readback after the run); the standup policy line reads `server_default_thinking=off request_thinking=on` on all 33 rows, as at the published tree.
- `scripts/sdfbase/math500/math500_thinking.sbatch` — E1's sbatch: `m5t_assert_launchable` before any write, the wave-isolated out root `controls_ext1/`, the recorder teardown; the served argv is the standup's.
- `scripts/sdfbase/math500/make_math500_overlays.py` — the overlay generator: `--check` byte-identity at ac67e138; adds the three new cells to each model's overlay (below).
- `scripts/sdfbase/math500/m5t_compat_preflight.py` — NEW: the P11 preflight tool itself (a reader; produces `compat_<key>.json`).
- `scripts/sdfbase/math500/m5t_dry_run.py` — NEW: the CPU dry run (prints the argv the launcher would compose; runs nothing).
- `scripts/sdfbase/math500/pull_wave.sh` — NEW: the read-only pull (an rsync include chain; no `--delete`).
- `scripts/sdfbase/math500/README.md` — documentation of the launcher.
- `scripts/sdfbase/full22/common.sh` — the shared standup composer (`f22_standup`, `f22_init`, the load check and the positive control): P11 read `server_default_thinking` from this file at BOTH trees (`git show <tree>:<launcher files>` for the published side, the dry run's STANDUP_POLICY line for the wave side) and found off == off; every launch flag it composes is in the echo the per-contrast check compares.
- `scripts/sdfbase/full22/load_check.py` — the standup's representative load check (S concurrent requests at the leg's p99 prompt + cap, greedy, cap 512): a preflight fired BEFORE the driver, whose requests are the untagged cap-512 greedy rows the per-kind recorder reports as the render kind's UNCLASSIFIED remainder; it sends no science request and changes nothing served.
- `scripts/sdfbase/full22/smoke_positive_control.py` — the steer-is-live positive control on lens cells (greedy bites at 48 on the cell and its `none__lens0` twin): a preflight, recorded in the ledger as `probe/*` rows, excluded from the ledger equality.
- `scripts/sdfbase/full22/positive_control_verdict.py` — the positive control's verdict helper (a reader).
- `scripts/sdfbase/full22/prompted_window_check.py` — the prompted route's window check (`assert_pass_fits_window`): refuses a prompt that does not fit; the constitution budgets fit with margin 699 / 780 / 793 at 24,064.
- `scripts/sdfbase/full22/served_thinking.sh` — the thinking-ON standup helper: the header certification before generation; header ids equal at both trees.
- `scripts/sdfbase/full22/roster_view.py` — the roster view reader (which cells a view names); the `main` view gains `prompted__si_constitution`.
- `scripts/sdfbase/full22/**` — the remaining full22 lane scripts (`bon_climb*`, `bon_served_run.py`, `judge/**`, `misalign/**`, `sampled/**`, `strength/**`, `petri_*`, `submit_*`, `target_*.sbatch`, `attacker_pool.sbatch`, `f22_roster.py`, `write_prompted_variant_dirs.py`, `wave_manifest.json`, READMEs): the jailbreak / judge / sampled-ladder / misalignment / dose lanes' drivers; none is on the MATH-500 launch path.
- `scripts/sdfbase/mt_serving_up.py` — the standup module (`mt_serving_up --thinking-on`; 9 commits): the header certification per template, the explicit serving window per leg, the `--enable-log-requests` opt-in, the `want` binding fix; the argv it composes is the echo compared per contrast, and the window is the row's (validated per row).
- `scripts/sdfbase/record_resolved_sampler.py` — the sampler recorder: a READER of the server log run after the leg; its verdict is recorded (`resolved_sampler.json`), never fed back into serving.
- `scripts/sdfbase/lora_serving_up.py` — the LoRA serving helper (bank views): the six LoRA legs' bank views are the published legs' (`make_bank_view.py`), objects pinned by sha.
- `scripts/sdfbase/multilora_*` — the native multi-LoRA admission / certification lane (its own sbatch and report); not on the MATH-500 path.
- `scripts/sdfbase/make_full22_configs.py` — the config generator for `configs/sdfbase_full22_*.yaml` (below): a generator whose outputs are reviewed by content.
- `scripts/sdfbase/build_probe_set.py` — the probe-set builder for the attacks lanes.
- `scripts/sdfbase/gsm8k/**` — the GSM8K paired-200 launcher (the other instrument).
- `scripts/sdfbase/mt/**` — the multi-turn attack sbatch (the jailbreak instrument).
- `scripts/capctrl_serving_up.py` — the serving-up helper `mt_serving_up.py` imports for the controls route: the readiness / endpoint checks; the served argv is the echo compared per contrast.
- `scripts/tinycap_oos_prompt_budget.py` — the prompt-budget tool: writes the `oos_prompt_budget.json` + `.fit.json` sidecars the launcher re-checks (`fits: true` on all six extension classes at ac67e138; the `none` class re-stamped with the same prompt maxima); admits `not_available` for llama (another instrument).
- `configs/sdfbase_full22_gemma4e4b.yaml` — the served gemma config: `prompted__si_constitution` added to `explicit_cells` (the new cell; the published legs' identity excludes `explicit_cells`), the jailbreak lanes' caps (`target_max_new_tokens` 512 → 4,096, `controls_max_new_tokens` 512, BoN climb keys) and the retired `wmdp` / `alpaca_eval` tasks — none of which MATH-500 reads (`max_tokens_by_task {inspect/math: 16384}` EQUAL, P11); `model` / `model_revision` unchanged (P11 `shared.model_id` / `model_revision` EQUAL).
- `configs/sdfbase_full22_qwen35_4b.yaml` — the served qwen config: the same three kinds of change as gemma's; model / revision unchanged (P11 EQUAL).
- `configs/sdfbase_full22_nemotron3nano4b.yaml` — the served nemotron config: the same three kinds of change; model / revision unchanged (P11 EQUAL).
- `configs/sdfbase_full22_glm46vflash.yaml` — the served GLM config: the same generator changes; GLM's published leg is read at 55e3cbdd only (no extension GLM leg; GLM is not in the panel and is excluded from the figures).
- `configs/sdfbase_full22_pins.json` — the object pins: 450 insertions, 0 deletions (dose-scaled `d*`, `it`-view and llama objects ADDED for other lanes); the six extension objects `base__constv2ctl2-full__lora_r4_ce` / `base__constv2-mix-full__lora_r4_ce` per panel model carry the SAME sha256 at both trees (8c0f6d97… / 8b2cd115… gemma, 4cd0f001… / 63c15165… qwen, b25435bd… / af46d415… nemotron), and each extension LoRA leg's `row_identity.object_artifact_sha256` equals its pin.
- `configs/sdfbase_full22_roster.yaml` — the roster: 454 insertions (the `prompted__si_constitution` cell, llama rows, sampled / dose views, the canonical attack caps) and 2 deleted COMMENT lines; the panel models' model / revision pins are unchanged (P11 EQUAL).
- `configs/sdfbase_full22_*.yaml` — every other generated config (the `_it` view, the dose-scaled `_d*`, the `_sampled_t*` and `_cap*` variants, `_halfdose`, the llama views): other lanes' served configs; no MATH-500 leg is served from any of them.
- `configs/sdfbase_transfer_*.yaml` — the transfer evaluation's configs; not served by MATH-500.
- `configs/sdfbase/math500/*.math500.yaml` — the MATH-500 overlays (gemma, qwen, nemotron, glm): the three new cells added to `explicit_cells`, plus the generator's attack-cap keys; `--check` byte-identity at ac67e138; the published `none` legs still pass the identity gate under these overlays because the identity excludes `explicit_cells` (R14) and the analysis reads the pulled manifest, not the overlay.
- `configs/sdfbase/gsm8k/**` — the GSM8K matrix, views and overlays (the other instrument).
- `configs/capeval/manifests/petri_full22_*.math500.yaml` — the MATH-500 served manifests (gemma, qwen, nemotron): the three new cells listed and `prompted_prefix_hashes[prompted__constitution]` = the SI's sha16 21e06f482a6559b3, verified per request (H8); nothing else in the manifest moved (the diff is those two hunks).
- `configs/capeval/manifests/**` — every other served manifest (Petri / GSM8K / misalign / alphasteer / llama / dose views): other instruments and views.
- `configs/capeval/mamba_backend_contract.json` — NEW: the nemotron fused-Mamba backend contract (the vLLM 0.25.1+cu129 wheel sha, the mixer / dispatch modules): pins the backend the attested environment already served the published nemotron legs with (`nemo_backend/mamba_backend_attestation.json` admitted + fused; the published nemotron logs show the same kernel overlay).
- `configs/capeval/tinycap-client-requirements.in` — a COMMENT-only change in the client requirements source; the compiled lock `tinycap-client-requirements.txt` is byte-identical at both trees (P11 `client_lock_sha256` EQUAL) and the venv is built from the lock.
- `configs/capeval/tinycap-harbor-requirements.*` — the Terminal-Bench harbor venv (the retired leg); not on the MATH-500 path.
- `configs/capeval/**` — the remaining capeval configs (`bcb_bfcl_200_widen_roster.yaml`, `domains.yaml`, `inspect_tasks/**`, `tasks/**`, `petri/**/BEHAVIOR.md`): other instruments' task registries and behaviour cards.
- `configs/sdfbase_sampled_decode.yaml` — NEW: the vendor sampler presets for the jailbreak and over-refusal drivers' sampled default; MATH-500's sampler comes from the row's pinned card in `configs/sdfbase/math500/sampling_rows.yaml` (unchanged between the trees) and the resolved sampler AGREES with the pinned row on every extension leg's 500 client requests; P11 `sampling_params` and `sampler_resolution` EQUAL.
- `configs/sdfbase_shipped_generation_config.json` — NEW: the shipped `generation_config.json` values pinned with provenance for the preset-vs-shipped test; a record, read by no server.
- `configs/tinycap/**` — the BigCodeBench / BFCL widen lane's configs.
- `configs/alphasteer_f22_*.yaml` — the AlphaSteer configs (its ladder; another wave's input, not this extension's).
- `configs/misalign_full22_*.yaml` — the misalignment-evals configs.
- `configs/two_si*.yaml` — the two-SI training / HPO configs (training side).
- `src/sisteer/data/instructions.py` — the SI registry: ADDS the FULL constitution variant (`si_constitution_full`, sha 82f88c55…) with its sha-checked loader; the SHORT SI `si_constitution` bytes are unchanged (sha256 21e06f48… asserted per request on the extension prompted legs and equal to the registry, P11 `arm_contracts.si`).
- `src/sisteer/data/constitution_full.md` — NEW: the vendored full-constitution text (another lane's cell, refused by name in the MATH-500 matrix).
- `src/sisteer/models.py` — the model registry (+90 lines: llama and other entries); the four served ids / revisions are unchanged (P11 EQUAL).
- `src/sisteer/methods.py` — the steering-method registry (training side).
- `src/sisteer/chat_markers.py` — NEW: chat-marker helpers used by XSTest / over-refusal controls and the header certification's marker lookups; the header ids are EQUAL at both trees.
- `src/sisteer/eval/**` — the attacks / controls / sampled-controls / sweep side (`attacks/**`, `cell_controls.py`, `sampled_controls.py`, `sampled_decode.py`, `derived_preflight.py`, `agentic_misalignment/**`, …): jailbreak, over-refusal and misalignment evaluation; not on the MATH-500 path.
- `src/sisteer/extraction/**` — the AlphaSteer / fit-meta extraction side (training).

## Non-producer paths (never run on the cluster for this instrument)

- `scripts/tinycap_math500_thinking.py` — the MATH-500 analysis itself (records its own `analysis_code_sha`; reads stores, produces none).
- `scripts/tinycap_publication_gate.py` — the publication gate (a reader of figure sidecars).
- `scripts/tinycap_gsm8k_*.py` — the GSM8K analysis and validators.
- `scripts/tinycap_ladder.py` — the ladder analysis helper.
- `scripts/tinycap_sandbox_watch.py` — the sandbox watcher for the BCB / BFCL legs.
- `scripts/alphasteer_f22/**` — the AlphaSteer scripts.
- `scripts/alphasteer_*.py` — the AlphaSteer screen / gate scripts.
- `scripts/modal/**` — the Modal launchers and the CPU suite.
- `scripts/sbatch/**` — other legs' sbatch files.
- `scripts/capeval_*.py` — the judged capability-eval analyses.
- `scripts/tb_*.py` — the Terminal-Bench pilot (retired leg).
- `scripts/bcb_bfcl_200_*.py` — the BigCodeBench / BFCL widen scripts.
- `scripts/bcbw_configs.py` — the BCB / BFCL widen configs generator.
- `scripts/bfcl_parser_probe.py` — the BFCL parser probe.
- `scripts/bench_eager_vs_graphs*` — the eager-vs-CUDA-graphs bench.
- `scripts/jailbreak_run.py` — the jailbreak driver (the attack scripts).
- `scripts/judge_*.py` — the judge clients and rubrics.
- `scripts/lens_gate_standup.py` — the lens gate standup (the lens scripts' own smoke).
- `scripts/lens_vectorize_gate.py` — the vectorised-hook gate.
- `scripts/mamba_backend_contract_author.py` — authors `configs/capeval/mamba_backend_contract.json` (a generator).
- `scripts/mamba_kernels_check.py` — the fused-kernel import check (a probe script; the launcher's own in-job guard is in `common.sh`).
- `scripts/petri_serving_gate.py` — the Petri serving gate.
- `scripts/two_si_run.py` — the two-SI training driver.
- `scripts/unified_sweep_run.py` — the sweep driver (training).
- `scripts/build_safety_defense.py` — the safety-defence builder (training data).
- `scripts/cap_ladder.py` — the cap-ladder CLI (GSM8K / sampled).
- `scripts/cell_controls_run.py` — the over-refusal controls driver.
- `scripts/compare_served_hf_task.py` — the served-vs-HF comparison probe.
- `scripts/fit_class_b.py` — the class-B fit (training).
- `scripts/glp_persona_anchor.py` — the GLP persona anchor scripts.
- `scripts/gsm8k_overlap_check.py` — the GSM8K overlap check.
- `scripts/backfill_fit_meta_object_sha256.py` — a one-off backfill of fit-meta object shas (writes fit-meta sidecars, not stores).
- `scripts/served_capability_judge.py` — the judged tiers' driver.
- `tests/**` — tests never run on the cluster.
- `docs/**` — documentation.
- `paper/**` — the paper notebook, figures and library.
- `results/**` — results records (prose; this analysis appends to one).
- `CLAUDE.md` — repo instructions.
- `NOTES.md` — notes.
- `evaluate.py` — the repository's top-level evaluate entry (training-side).
- `pyproject.toml` — the development-machine project metadata; the cluster venvs are built from the `configs/capeval/*-requirements.txt` locks, which are unchanged.
- `uv.lock` — the development-machine project lock; the cluster venvs are built from the `configs/capeval/*-requirements.txt` locks, which are unchanged.
