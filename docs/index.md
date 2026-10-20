# Documentation

Reference pages for the code and data behind *Constitutional adapters: inference-time interventions for misalignment
and misuse*. Start with the repository [README](../README.md); to run the pipeline on another model, start with
[reproduce/README.md](../reproduce/README.md). These pages hold the detail those two point at.

| page | what it answers |
|---|---|
| [`configuration.md`](configuration.md) | the site config, the cluster blocks, `reproduce/models.yaml`, run overrides |
| [`secrets.md`](secrets.md) | how credentials reach the scripts, and what never lands on a cluster |
| [`registry/`](registry/README.md) | the results registry every number is read from: concepts, snapshot, schema, records, vocabulary |

The figures have [paper/README.md](../paper/README.md) and [paper/notebooks/README.md](../paper/notebooks/README.md)
(notebook, snapshot, figure map, nbconvert route). The data have
[corpus_generator/README.md](../corpus_generator/README.md), [system_instructions/README.md](../system_instructions/README.md),
[objects/README.md](../objects/README.md), [data/README.md](../data/README.md), [configs/README.md](../configs/README.md),
and each corpus part's `MANIFEST.json` under `corpora/`.

The other pages are design notes. These bear on reading a result:

| note | the fact it establishes |
|---|---|
| [`static_padding.md`](static_padding.md) | the padding ladder: neutral filler tokens added to single-turn jailbreak prompts |
| [`cap_ladder.md`](cap_ladder.md) | the truncation rule: capped responses re-run at larger caps until the truncated share is at or below 1% or 16,384 tokens |
| [`served_geometry_p99.md`](served_geometry_p99.md) | the measured request geometry behind every served eval's context window and batch size, the multi-turn attacks' 4,096-token reply cap among them |
| [`tinycap_interfaces.md`](tinycap_interfaces.md) | the capability harness's contracts: GSM8K as a paired 200-item subset of inspect_evals' stock task, the samples drawn per item, the control slots |
| [`scripts/sdfbase/full22/README.md`](../scripts/sdfbase/full22/README.md) | best-of-N: a 512-token store, then a 4,096-token re-draw of its capped draws at the same per-draw seed |

The rest are engineering notes, none on the reproduction path: [`attacker_pool.md`](attacker_pool.md) (the shared
multi-turn attacker served on Modal and its window sizing), [`served_thinking_on.md`](served_thinking_on.md) (serving
fitted objects under a thinking-on template) and
[`lens_stacked_reductions_scoping.md`](lens_stacked_reductions_scoping.md) (the stacked-reduction contract of the
hook-based lens engine).
