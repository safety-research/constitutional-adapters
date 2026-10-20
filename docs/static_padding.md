# Neutral long-context padding for the static families

The padding knob asks whether a static one-shot attack lands differently when
the user turn is long. It adds a measured number of neutral PG-19 filler
tokens to every item's user turn, before or after the family's transformed
prompt, and leaves the transform, decode, cap and store layout unchanged, so
growing the token count grows only the padding while the payload stays
byte-identical. Code: `src/sisteer/eval/attacks/static_padding.py` (the
contract), `static_common.run_static` (applies it), `scripts/jailbreak_run.py`
(the CLI), `scripts/fetch_stage2_data.py --only pg19-filler` (the corpus),
`scripts/sdfbase/full22/static_padding/` (the wave).

## Scope

The four static one-shot families `pasttense`, `codeattack`, `renellm`,
`grok420` (`jailbreak_run._SAMPLED_STATIC_FAMILIES`). Padding requested for
any other attack, or on the XSTest pass, refuses. Served and HF targets both
pad; on the served path only a tokenizer loads, at the config's `model` /
`model_revision` pins.

## The surface

Off unless a positive token count reaches the run; with none, every byte the
static path writes is as before.

| flag | config key (`attacks.static_padding.*`) | default |
|---|---|---|
| `--static-padding-tokens N` | `tokens` | 0 = off |
| `--static-padding-placement {before,after}` | `placement` | `before` |
| `--static-padding-mode {fixed,per_goal}` | `mode` | `fixed` |
| `--static-padding-filler PATH` | `filler_cache` | searched (below) |
| `--static-padding-tolerance F` | `tolerance` | 0.02 |
| `--static-padding-reserve N` | `reserve` | 2048 |

A typed flag wins over the config block; the block's keys are closed; a
padding sub-flag typed while the knob is off refuses. `tokens` is the padding
added, never a total prompt length. `fixed` pads every item with the same
filler prefix; `per_goal` takes a deterministic per-goal offset into the
filler stream.

## Store identity and the row record

Padded and unpadded rows never share a store: `run_static` refuses to append
rows under a different padding block, so a padded run gets its own
`private_run` (`<base>_pad16384`). Every row keeps `variant` /
`attack_prompt` as the unpadded text and gains a `padding` block: the
identity keys `schema, tokens, placement, mode, source, filler_sha256,
separator, tolerance` plus the per-item record `pad_tokens_target`,
`pad_tokens_actual` (within `tolerance`, or the item raises),
`filler_offset_tokens` and `padded_prompt_sha256`. The progress record
carries `padding_tokens`; `served_target.json` carries `static_padding`. The
rung driver `scripts/sdfbase/full22/sampled/escalate_static.py` refuses a
padded base: it re-sends `attack_prompt`, which would re-draw the items
without their padding.

## The window preflight

`tokens + attacks.static_max_new_tokens + reserve` must fit
`--target-max-model-len`; the check runs before any GPU work and names all
four numbers. The library reserve of 2,048 stands in for instruction +
payload + chat template; the wave pins `reserve: 8192` because the
constitution instruction alone renders to about 6,000 tokens, and its
preflight (`padding_preflight.py`) derives the real need by rendering every
variant of every goal under each cell's instruction with the pinned
tokenizer. The wave serves every level at one window, 98,304, so the five
levels share one geometry.

### The cap is the ladder's ceiling, so the wave runs once

Every level's config, the unpadded control included, sets
`attacks.static_max_new_tokens` to 16,384, the static ladder's ceiling
(`make_padding_configs.STATIC_CAP == ladder_decide.CEILING`; the preflight
refuses any other cap). The leg is prefill-bound, so a truncation rung would
re-pay the whole pad prefill per re-drawn item, and the rung driver refuses
padded stores anyway; one pass at the ceiling means no rung can be needed.

## Where the filler has to be

The corpus is the PG-19 cache the long-context suite uses: a JSON object with
a `"text"` key, 2,881,719 bytes, sha256
`7351d518dc71c53bfc0637da825db82e1d7dc4c04e9acd0e8f015ea8d7826b97`
(`padding_preflight.PG19_FILLER_SHA256`). It is fetched, never tracked:

```bash
python scripts/fetch_stage2_data.py --only pg19-filler [--filler-out PATH]
```

Resolution order when `--static-padding-filler` is omitted: the config's
`filler_cache`, `$SISTEER_PG19_FILLER`, `<repo>/data/pg19_filler.json`, then
the same relative path under a shared storage root. A named path that does
not exist refuses; the recorded `filler_sha256` pins the corpus either way.

## The padded-static wave

`make_padding_configs.py` renders each level's config and manifest from the
campaign originals, changing only the named keys:

| | |
|---|---|
| levels (`PADS`) | 0, 8,192, 16,384, 32,768, 65,536 tokens |
| placement / mode / reserve / cap | `before` / `fixed` / 8,192 / 16,384 |
| configs | `configs/sdfbase/static_padding/sdfbase_full22_<key>.padstatic[_obj[_s<N>]]_p<pad>.yaml` |
| manifests | `configs/capeval/manifests/petri_full22_<key>.padstatic[_obj[_s<N>]].yaml` |
| stores | `pad_static_<key>[_obj[_s<N>]]_p<pad>` |
| jobs | `target_static_padded.sbatch` (prompted-only cells), `target_static_padded_obj.sbatch` (steered cells); one job per (level, family) |

### The steered ten-cell wave

The `_obj` family serves ten cells at every level: `none`, the two prompted
instruction baselines, the five fitted seed-1 objects of the production
roster's `s1` view, the all-positions Arditi rung `arditi_allpos` and the
full-dose paired write `constv2-mix-full__psr_ce_all_minus_ctl`
(`make_padding_configs.OBJ_CELLS`).

### Seed replicates

The `obj_s2` / `obj_s3` families (`OBJ_SEED_FAMILIES`) re-run the six fitted
objects at training seeds 2 and 3 over the same five levels, seven cells per
seed (`OBJ_SEED_CELLS`): a same-wave `none` anchor plus the six seed-N
objects; no prompted baseline and no Arditi rung, which have no training
seed. Their rosters list fitted objects only.

## Grading

`grade_padded_static.py` grades one (level, family) with the campaign's
StrongREJECT instrument through the shared body
`src/sisteer/eval/static_grading.py`; `grade_padded_family.sh` mirrors a
store and calls it (`--check-only`, `--dry-run`,
`--config-family obj|obj_s2|obj_s3`). The lane holds every store to its
padding block before any spend (`kind: base`, `cap: 16384`, no rung fields;
the unpadded control the mirror image) and asserts the judge never sees the
filler: items are built from the harness goal text and the stored `response`
alone, and any slot hashing to a row's `padded_prompt_sha256` refuses.
Outputs land under `<grades-root>/pad_static_<key>_p<pad>/<family>/` with
`pad_tokens` and the padding block on every cell record.
