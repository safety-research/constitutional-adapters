# Thinking-ON serving (`mt_serving_up.py --thinking-on`)

Every fitted object was fit under thinking-OFF, one-turn renders; a thinking-ON standup serves those
same objects out of support, as a test. On three of the four model templates the assistant header
moves with the render, so the served header is declared per template, certified offline before any
GPU minute, and attested back by every worker. Default OFF is the ordinary standup byte for byte.

## The flag

`scripts/sdfbase/mt_serving_up.py ... --thinking-on` (store_true; default OFF). In order:

1. **Launch.** `build_server_launch(..., thinking_on=True)` composes
   `--default-chat-template-kwargs '{"enable_thinking": true}'` (`vllm_server.THINKING_ON_TEMPLATE_KWARGS`)
   in place of the OFF pin, stated explicitly even where the template's own default is ON; refused on a
   template with no thinking switch. No `--reasoning-parser` is composed: the completion carries the
   raw `<think>…</think>` block and the GSM8K extractor splits on `</think>` client-side.
2. **Certification** (`chat_markers.thinking_on_certification(tok)`, on the served tokenizer under the
   OFF pin): `off_ids` (the OFF header), `on_ids` (what the ON render's `add_generation_prompt` appends
   after the user turn), the checks `clean_suffix`, `thinking_turn_prefix` (required), `plain_turn_prefix`
   (recorded; `chat_markers.THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX` makes it required),
   `off_header_absent_in_on_render` (required where the header moves), `tail` (required), and a `verdict`
   ∈ `certified` / `trivially_in_support` (`on_ids == off_ids`) / `no_switch` / `refused` / `error`.
   `--thinking-on` refuses the standup (`SystemExit`, before any launch) on `no_switch` / `refused` / `error`.
3. **Declaration by name.** Every lens cell whose served ops are header-located (`positions="response"`,
   classified by adapting the cell's bytes exactly as the worker will, `mt_serving_up.lens_header_dependence`)
   is respelled `<cell>@thinking` (`mt_serving_up.declared_cell_spellings`), so the overlay
   `serve_out_of_support: ["thinking"]` is applied by `registry_adapter` and hashed into the intervention
   hash; the vmodel is `petri__<cell>@thinking`. The certified `on_ids` are exported into the registry
   (`lens_steer.REGISTRY_THINKING_HEADER_KEY` = `assistant_header_ids_thinking_on`) and attached to every
   response op of every declared cell (`lens_adapters.attach_thinking_header`); the `none__lens0` twin
   inherits its donor's declaration. A cell served at every position (`positions=all`) has no header to
   relocate: served undeclared, recorded as `thinking_header_dependence[cell] == "all"`. OFF classifies
   no mask (`resolve_thinking(thinking_on=False)` records `{}`).
4. **Worker cross-check + attestation.** A registry whose loaded ops carry the second header runs
   `lens_steer.verify_assistant_header_thinking_on` at preflight (the header re-derived from the served
   tokenizer under the plain-turn rule must equal the registry's; gemma-4's equal ON/OFF header is
   admitted). Every install report carries `assistant_header_ids_thinking_on` and `lens_serve_declarations`;
   the standup reads them with `lens_worker.thinking_header_attested` / `serve_declarations_attested`
   and `thinking_attestation_problems(attestation, mode=, want_header=, expected_decls=)`, refusing
   to serve unless they equal what it exported (`expected_declarations(entries)`).
5. **Proxy.** `RoutingProxy(thinking_mode="on", thinking_header=<fact>, lens_serve_declarations=<map>)`.
   Every lens-route request must carry `petri_thinking_on: true` (`proxy.THINKING_ON_KEY`); the key never
   travels upstream. The selector fact `thinking_on` (`vllm_xargs`; the third fact beside `response_prefill`
   / `agent_loop`, mutually exclusive) is injected only for a declared cell on a template whose switch
   moves the header; elsewhere the declaration is validated and recorded, not injected.

## The proxy's refusal codes (`contracts.THINKING_REFUSAL_CODES`, nine)

| code | when |
|---|---|
| `thinking_kwargs_invalid` | `chat_template_kwargs` not an object, a foreign key, or a non-bool value (any route) |
| `thinking_kwargs_inert` | `enable_thinking: true` on a template with no thinking switch |
| `thinking_on_invalid` | `petri_thinking_on` present and not a JSON boolean (a present `null` included) |
| `thinking_on_mismatch` | declared beside `petri_response_prefill` / `petri_agent_loop` on a template not certified moves-false, or declared without `enable_thinking: true` |
| `thinking_on_not_lens` | the field present on a non-lens route, whatever its value |
| `thinking_on_undeclared` | ON standup: any lens-route request without `petri_thinking_on: true`, whatever the cell's served positions (an omitted kwarg renders ON). OFF proxy: `enable_thinking: true` on a lens route with no declaration when the cell's served mask is header-located (`response`) or the routing table's served-positions map does not name the cell; a cell the map states as `all` / `last` only is accepted, nothing injected |
| `thinking_on_unattested` | a declaration with no attested `lens_serve_declarations` map or no `thinking_header` fact to verify against |
| `thinking_on_cell_not_declared` | OFF proxy: a declaration on a moves-true template whose cell's attested declaration lacks `thinking`. ON standup: not raised — such a cell is an all-position fit the standup left undeclared, so the declaration is validated and recorded, nothing injected |
| `thinking_regime_mismatch` | `enable_thinking: false` (chat or tokenize) on a standup attested ON |

`enable_thinking: false` / absent is accepted everywhere on an OFF proxy. An ON proxy stamps
`render_regime: "thinking_on"` on every forwarded request's audit row and in the response's `petri`
block; an OFF proxy's rows and responses are unchanged.

## Fields (exact names)

`endpoint.json` and `cells_served.json` (both, via `mt_serving_up.thinking_record`):

| field | value |
|---|---|
| `thinking_mode` | `"off"` / `"on"`; always present |
| `thinking_declaration_key` | `"petri_thinking_on"` |
| `default_chat_template_kwargs` | the composed launch pin string (`{"enable_thinking": true}` / `false`); null before launch |
| `thinking_header` | `{switch, moves, off_ids, on_ids}` (the proxy's fact; `on_ids` / `moves` null when uncertified) |
| `thinking_header_error` | why `on_ids` is null on a switched template, else null |
| `thinking_certification` | the full certification dict (+ `model_key`, `off_header_len`, `on_header_len`, `requires_plain_turn_prefix`) |
| `thinking_declared_cells` | harness ids declared `serve_out_of_support: ["thinking"]` (`[]` on OFF / all-fit waves) |
| `thinking_header_dependence` | lens cell id → `"response"` / `"all"` / `"<mode>+<mode>"` |
| `assistant_header_ids_thinking_on` | the ON header exported into the registry, or null |
| `attested_assistant_header_ids_thinking_on` | what every worker attested, or null; `_error` twin beside it |
| `lens_serve_declarations` | the workers' attested cell → declaration map (`{}` = attested none); `lens_serve_declarations_error` beside it |

`/v1/models` → `petri_proxy`: `thinking_mode` (always present), `thinking_declaration_key`,
`thinking_header` / `thinking_header_error`, `lens_serve_declarations` / `lens_serve_declarations_error`.
Routing table (`routing_table.json`): top-level `thinking_header` + `thinking_header_error`; a proxy
rebuilt from the table via `proxy.main --thinking-mode on --lens-serve-declarations <json>` holds the
same fact. Registry index (`lens_registry/registry.json`): `assistant_header_ids_thinking_on` only when
declared cells exist. Install report (`steering_attestation.json` → `worker_reports[*]`):
`assistant_header_ids_thinking_on` and `lens_serve_declarations`.

## Per-model certification (the four templates, pinned tokenizers)

| model key | OFF header (H) | ON header (H) | moves | verdict |
|---|---|---|---|---|
| `qwen35_4b` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` (7: 248045 74455 198 248068 271 248069 271) | `<\|im_start\|>assistant\n<think>\n` (5: 248045 74455 198 248068 198) | yes | certified |
| `gemma4e4b` | `<\|turn>model\n` (3: 105 4368 107) | same (3) | no | trivially in support — the ON render adds a leading `<\|turn>system\n<\|think\|>\n…` block; the header is unchanged, nothing is declared |
| `nemotron3nano4b` | `<\|im_start\|>assistant\n<think></think>` (6: 10 1503 19464 1010 12 13) | `<\|im_start\|>assistant\n<think>\n` (6: 10 1503 19464 1010 12 1010) | yes (last token) | certified by the thinking-turn rule; `plain_turn_prefix: false` recorded (the empty-reasoning completed turn collapses to `<think></think>`), so a declaring header-dependent cell refuses at preflight on this family |
| `glm46vflash` | `<\|assistant\|>\n<think></think>\n` (5: 151337 198 151350 151351 198) | `<\|assistant\|>\n` (2: 151337 198) | yes | certified; the OFF render also appends `/nothink` to the user turn |

The kwarg that drives every switch is `enable_thinking`; `{"enable_thinking": true}` selects the ON branch
on all four. Two ON-header rules are in force: the registry's header (the plain-turn rule, cross-checked
by every worker) and the standup-wide certification (the thinking-turn rule, with the plain-turn
verdict recorded beside it); where both derive, the standup refuses if they disagree.

## Byte-identity (thinking-OFF)

Pinned by `tests/test_thinking_on_standup.py` / `tests/test_thinking_on_proxy.py`: the launch argv/env with
`thinking_on=False` is the ordinary composition; an undeclared registry's `registry.json` has no new key;
the OFF proxy's upstream body, audit row and response are unchanged; the OFF derivation
`generation_header_ids_under(tok, enable_thinking=False)` calls `generation_header_ids` under the OFF pin.

## Caveats

- `registry_shim.build_lens_registry` (the capeval shim path) takes no ON header: a declaring condition
  there refuses at worker load (`attach_thinking_header`: no header supplied) — fail closed.
- A spec carrying `serve_experiment` position variants refuses.
- Agent-loop / seeded renders under thinking-ON are refused on templates whose header moves
  (`thinking_on_mismatch`).
