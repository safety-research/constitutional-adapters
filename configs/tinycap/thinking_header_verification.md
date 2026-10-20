# Thinking-header verification: does serving thinking-ON move the response mask a fitted object was derived under?

Measured offline through each model's pinned tokenizer (the revision each campaign config pins); the raw ids are in
`thinking_header_measure_2026-09-09.json` beside this file.

## Method

Render a one-user-message chat with `add_generation_prompt=True` and `=False`, under `enable_thinking=False` and
`=True`, and take the token difference. That difference is `response_header_len` (H), the trailing assistant header the
response mask keys on. A fitted object is served correctly only if the header's width and ids at serve time equal those
at fit time.

## Result

| model | template carries `enable_thinking` | H OFF | H ON | header text | mask boundary moves? |
|---|---|---|---|---|---|
| gemma-4-E4B-it, gemma-4-E2B-it | yes | 3 | 3 | `<\|turn>model\n` (both) | no |
| Qwen3.5-9B, Qwen3.5-4B | yes | 7 | 5 | OFF `<\|im_start\|>assistant\n<think>\n\n</think>\n\n`; ON `<\|im_start\|>assistant\n<think>\n` | yes (7 → 5) |
| Llama-3.1-8B-Instruct | no | 4 | n/a | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` | no channel at all |
| GLM-4.6V-Flash | yes | 5 | 2 | OFF `<\|assistant\|>\n<think></think>\n`; ON `<\|assistant\|>\n`; OFF also appends the control token `/nothink` to every user turn | yes (5 → 2) |
| Nemotron-3-Nano-4B | yes | 6 | 6 | OFF `<\|im_start\|>assistant\n<think></think>`; ON `<\|im_start\|>assistant\n<think>\n` | width no, ids yes: the last token changes `</think>` (13) → `\n` (1010) and ON leaves the block open |

| model | OFF ids | ON ids | delimiters |
|---|---|---|---|
| GLM-4.6V-Flash | `[151337, 198, 151350, 151351, 198]` | `[151337, 198]` | `<think>` 151350, `</think>` 151351 (added, not control); `/nothink` 151360 (control) |
| Nemotron-3-Nano-4B | `[10, 1503, 19464, 1010, 12, 13]` | `[10, 1503, 19464, 1010, 12, 1010]` | `<think>` 12, `</think>` 13 (added, not control); `<\|im_end\|>` 11, closing scaffold `[11, 1010]` |

Template sha256[:16]: GLM `bf6cfe131f9af1cd` (the `template_hash` the GLM fitted objects recorded), Nemotron
`ab7813c3abdd9cb6`. The Qwen ids are `QWEN35_HEADER_IDS_OFF` / `_ON` in `sisteer.capeval.tinycap.contracts`.

## What it means

1. `models.pin_reasoning_template_off` fires whenever the template mentions `enable_thinking`, gemma-4's included, so
   every fitted object was derived under a thinking-OFF render. Llama has no switch and is OFF by property.
2. Gemma-4 signals thinking with a leading system turn (`<|turn>system\n<|think|>\n<turn|>`, 7 tokens at the front) and
   leaves the trailing header identical: the mask is invariant, only the input distribution changes.
3. Qwen: OFF puts an empty `<think></think>` block inside the header (H = 7), ON leaves it open (H = 5); an object fit at
   7 served at 5 masks two tokens of the model's own thinking as response. Refuse.
4. GLM moves 5 → 2 in the opposite shape: the ON header carries no think tag (the completion emits both tags) and is a
   strict prefix of the OFF header; under OFF every fit prompt also ended `…/nothink<|assistant|>\n<think></think>\n`.
5. Nemotron is the case a width-only check cannot see: H is 6 in both modes, yet the last header token differs and the
   ON prompt ends in an open `<think>\n`, so the reasoning lands inside the response mask. It must not join
   `HEADER_INVARIANT_THINKING_FAMILIES`. The id-level derivation (`_appended_generation_header`'s prefix rule) refuses
   its ON header, `thinking_header_facts_or_error` records `on_ids: None, moves: None`, and the proxy refuses
   `thinking_on_unattested`: fail closed.

## Which switch the templates honour

`enable_thinking` is the only honoured kwarg on Qwen3.5, GLM and Nemotron, and all three default to thinking-ON when
it is absent. An unrecognised kwarg (`thinking=False`, `enable_reasoning=False`) is ignored → ON; the string `"false"`
is truthy → ON; `None` and `0` give OFF on GLM and Nemotron (truthiness) but ON on Qwen3.5 (`is false`). Gemma's header
is identical under every probe. Pass a JSON boolean, never a string; the served side's `THINKING_OFF_TEMPLATE_KWARGS =
'{"enable_thinking": false}'` is one. Nemotron's `truncate_history_thinking` affects only historical assistant turns.

## Not verified here

What the models emit under ON (that GLM opens with `<think>`, that Nemotron closes the prompt-opened block) is
template-implied, not observed: this measurement is CPU-only. These are the HF tokenizer's renders; the live vLLM
`/v1/tokenize` render (`measure_thinking_header_by_arm`, source `served_tokenize`) is the authoritative serve-time
reading and the preflight's job, tied back to this one by the template hashes above. The GLM and Nemotron OFF ids are
pinned by `tests/test_glm_chat_markers.py` and `tests/test_nemotron_chat_markers.py`.
